import numpy as np 
import pandas as pd 
import datetime 
import pickle
import os
import time
import sys
from collections import OrderedDict
from scipy.stats import gamma, lognorm
import textwrap
from datetime import timedelta
from io import StringIO

import pymc3 as pm
from scipy.linalg import toeplitz
import theano.tensor as tt

%matplotlib inline
import matplotlib.pyplot as plt  
import seaborn as sns
########################################
###        HELPER FUNCTIONS      #######
########################################
def simple_smoother( s, window_size = 5):
    
    neg_idxs = [ i for i in range(len(s)) if s.iloc[i] < 0 ]
    if not neg_idxs:
        return s, False
    else:
        for i in neg_idxs:
            var_later = np.std(s.iloc[i+1:i+1+window_size].values )/ np.mean(s.iloc[i+1:i+1+window_size].values )
            var_earlier = np.std(s.iloc[i-window_size:i].values )/  np.mean(s.iloc[i-window_size:i].values )
            if var_later > var_earlier:  
                interval = (i, i+window_size+1) 
                tot = s.iloc[interval[0]: interval[1]].sum()
                s.iloc[i] = tot/ (window_size + 1)
                s.iloc[interval[0]+1:interval[1]] =(window_size/(window_size+1))*tot*( 
                                                        s.iloc[interval[0]+1:interval[1]]/(s.iloc[interval[0]+1:interval[1]].sum() ) )
            else:
                interval = (i - window_size, i+1 )
                tot = s.iloc[interval[0]: interval[1]].sum()
                s.iloc[i] = tot/ (window_size + 1)
                s.iloc[interval[0]:interval[1]-1] =(window_size/(window_size+1))*tot*( 
                                                        s.iloc[interval[0]:interval[1]-1]/(s.iloc[interval[0]:interval[1]-1].sum() ) )
        return s.astype(int), True
    
def load_JHU_deaths(f_i = None, agg_by_state = True):
    """Fetch cumulative deaths from JHUs Covid github with optional aggregation by state. Note: this is not number of deaths occuring on each day
    
    Keyword Arguments:
        f_i {[type]} -- [description] (default: {None})
        agg_by_state {bool} -- [description] (default: {True})
    
    Returns:
        [type] -- [description]
    """
    
    if f_i is None:
        f_i = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
    df = pd.read_csv(f_i)
    #df = df.loc[df.iloc[:,5] != "Unassigned", : ].copy()
    cols = [6]
    cols.extend(np.arange(12, df.shape[1]))
    if agg_by_state:
        df = df.groupby("Province_State").apply(lambda x : x.iloc[:, cols[1:]].sum(axis = 0))
    return df

def plot_shaded(data, figsize = (12,5), **kwargs ):
    """
    data - df -- index is x values, 
                 columns is [mean, upper ,lower]. If columns is multilevel level 0 stores names of different trances
                        level 1 stores mean, upper, lower
                 data is y values
    """
    
    fig, ax = plt.subplots(figsize = figsize)
    
    if isinstance(data.columns, pd.MultiIndex):
        for l in data.columns.levels[0]:
            data.loc[: , (l, "mean")].plot(ax = ax, label = l, marker= 'o')
            x_data = ax.get_lines()[-1].get_xdata()
            ax.fill_between(x_data,  
                            data.loc[: , (l, "lower")].values ,
                            data.loc[: , (l, "upper")].values ,
                           **kwargs)
    ax.legend()
    return fig

def make_infect_to_death_pmf( params_infect_to_symp = (1.62, 0.42),  params_symp_to_death = (18.8,  0.45), 
                             distrib_infect_to_symp = "lognorm" ,  distrib_symp_to_death = "gamma"):
    """Construct numpy array representing pmf of distribution of days from infection to death. 

    Keyword Arguments:
        params_infect_to_symp {tuple} -- [description] (default: {(1.62, 0.42)})
        params_symp_to_death {tuple} -- [description] (default: {(18.8,  0.45)})
        distrib_infect_to_symp {str} -- [description] (default: {"lognorm"})
        distrib_symp_to_death {str} -- [description] (default: {"gamma"})

    Returns:
        [type] -- [description]
    """
    def make_gamma_pmf(mean, cv, truncate = 4):
        rv = gamma(a = 1./(cv)**2, scale = mean*(cv)**2)
        lower = 0 
        upper = int(np.ceil( mean + cv*mean*truncate) )
        cdf_samples = [rv.cdf(x) for x in range(lower, upper + 1, 1)]
        pmf = np.diff(cdf_samples)  
        pmf  = pmf / pmf.sum() ## normalize following truncation
        return pmf
        
    def make_lognorm_pmf( log_mean, log_sd, truncate=4):
        rv = lognorm( s = log_sd, scale = np.exp(log_mean),  )
        lower = 0 
        mean, var = rv.stats(moments = 'mv')
        upper = int(np.ceil( mean + np.sqrt(var)*truncate) )
        cdf_samples = [rv.cdf(x) for x in range(lower, upper + 1, 1)]
        pmf = np.diff(cdf_samples)  
        pmf  = pmf / pmf.sum() ## normalize following truncation
        return pmf
        
    if distrib_infect_to_symp.lower() == "lognorm":
        pmf_i_to_s = make_lognorm_pmf(*params_infect_to_symp)
    else:
        raise NotImplementedError()
    if  distrib_symp_to_death.lower() == "gamma":
        pmf_s_to_d = make_gamma_pmf(*params_symp_to_death)
    else:
        raise NotImplementedError()
    
    pmf_i_to_d =  np.convolve(pmf_i_to_s, pmf_s_to_d )
    ## truncate and renormalize
    cdf = np.cumsum(pmf_i_to_d)
    min_x = np.argmax( cdf >= 0.005  )  
    max_x = len(cdf) - np.argmax( (1. - cdf[::-1]) >= 0.005 ) 
    pmf_i_to_d[:min_x] = 0.
    pmf_i_to_d = pmf_i_to_d[: max_x]/ np.sum(pmf_i_to_d[: max_x])
    
    return pmf_i_to_d
#####################################
###         CLASSES      ############
#####################################
class Infection_Series_Estimator(object):
    """
    Properties:
            P_e_given_i - (2d array) P_e_given_i[e,i] gives probability of event on day e given infection on day i.
            T_period - (2 tuple) — (min_days_elapsed_from_infect_to_transmission, max_days_elapsed_from_infect_to_transmission)
                                       (lower end inclusive, upper end exclusive)
            mu — exponential prefactor in prior
            inputs — dictionary of inputs
            D - Matrix for calculating 1d discrete difference in log(tranmission_rataes)        
    """
    def __init__(self, event_series, P_e_given_i, T_period, policy_dates):
        """ 
        Inputs:
            event_series — pd.Series - index is dates values are counts
            P_e_given_i - 1d array repreenting pmf for event occuring k = (e-i) days after infection
                        
            T_period - (2 tuple) — (min_days_elapsed_from_infect_to_transmission, max_days_elapsed_from_infect_to_transmission)
                                       (lower end inclusive, upper end exclusive)
            rms_foldchange — root mean square log2 ratio of tranmission rates on sequential dates. Determines strength of pior
        """
        self.init_kwargs = { "event_series" :event_series,
                        "P_e_given_i": P_e_given_i,
                       "T_period" : T_period,
                       "policy_dates" : policy_dates}
        ## counts and date info
        self.event_series = self._preprocess_event_series(event_series, P_e_given_i)
        date_max = self.event_series.index[-1]
        date_min = self.event_series.index[0]
        self.days_total = (date_max - date_min).days + 1
        
        if T_period[0] != 1:
            raise ValueError("Code only supports models with transmission period beginning 1 d after infection")
        self.T_period = T_period
        
        policy_dates_processed = self._preprocess_policy_dates(policy_dates)
        self.policy_dates = [ ( (x[0]-date_min).days, (x[1]-date_min).days) for x in  policy_dates_processed ]
        
    @staticmethod
    def _preprocess_event_series(event_series, P_e_given_i_1d):
        """
        Insures event series is:
            - sorted by increasing date
            - has timestamp index
            - starts at at least len(P_e_given_i_1d) -1 before first day with event_series >0
        
        Returns
             event_series
        """
        ## Sort and check index type
        event_series = event_series.sort_index().copy()
        if not isinstance(event_series.index, pd.core.indexes.datetimes.DatetimeIndex):
            event_series.index = pd.to_datetime(event_series.index)
        ## Truncate or extend index
        earliest_nonzero = (event_series > 0).idxmax() 
        date_min = earliest_nonzero - pd.Timedelta(len(P_e_given_i_1d)-1, unit = "days")                  
        if date_min < event_series.index[0]:
            prepend_series =  pd.Series(index = pd.date_range(start = date_min,
                                                              end =event_series.index[0],
                                                               freq = 'D', closed = 'left' ),
                                        data = 0)     
            event_series = pd.concat( [prepend_series, event_series ] )    
        else:
            event_series = event_series.loc[date_min:].copy()
        return event_series
    
    @staticmethod
    def _preprocess_policy_dates(policy_dates):
        out = []
        for elem in policy_dates:
            if isinstance(elem, str):
                out.append( ( pd.Timestamp(elem), pd.Timestamp(elem) + pd.Timedelta(1, unit = "D") ) )
            elif isinstance(elem, tuple) or isinstance(elem, list):
                if len(elem) == 1:
                    out.append( ( pd.Timestamp(elem[0]), pd.Timestamp(elem[0]) + pd.Timedelta(1, unit = "D") ) )
                elif len(elem) == 2:
                    out.append( tuple( pd.Timestamp(x) for x in elem ) )
                else:
                    raise ValueError("At least one of the entries of policy_dates has length > 2!")
            else:
                raise ValueError("Could not parse entry {} of policy_dates".format(elem ))
        return out
    
    @property
    def P_e_given_i(self):
        P_e_given_i = self._make_P_e_given_i(self.init_kwargs["P_e_given_i"], self.days_total, observed_only = True)
        return  P_e_given_i
    @property
    def M1(self):
        return None
    @property
    def M2(self):
        return None
    @property
    def D(self):
        return None
    
    @staticmethod
    def _make_P_e_given_i(P_e_given_i, days_total, observed_only = True):
        """
        P_e_given_i - 1d numpy array
        """
        if observed_only:
            P_e_given_i = toeplitz(c = np.concatenate([ P_e_given_i, np.zeros(days_total-len(P_e_given_i)) ] ),
                                   r =  np.concatenate([ P_e_given_i[0:1], np.zeros(days_total- 1) ] ), 
                                  )
        else:
            P_e_given_i = toeplitz(c = np.concatenate([ P_e_given_i, np.zeros(days_total-1) ] ),
                                   r =  np.concatenate([ P_e_given_i[0:1], np.zeros(days_total- 1) ] ), 
                                  )
        return P_e_given_i 
  
    @staticmethod
    def _make_pior_mats(T_period, days_total):
        """
        
        """
        M1 =  toeplitz( c = np.zeros( days_total-(T_period[1]-T_period[0]) ) ,
                        r = np.concatenate([ np.zeros(T_period[1]-T_period[0]),
                                             np.array([1.]) , 
                                             np.zeros( days_total - 1 - (T_period[1]-T_period[0]) ), 
                                           ])
                      )
        M2 =  toeplitz( c = np.concatenate([ np.array([1.]), np.zeros( days_total-(T_period[1]-T_period[0]) -1 ) ] ) ,
                        r = np.concatenate([ np.ones(T_period[1]-T_period[0]),
                                            np.zeros(days_total - (T_period[1]-T_period[0])) ] )
                      )
        assert M1.shape == M2.shape
        D =  toeplitz( c = np.concatenate([ np.array([-1.]), np.zeros(M1.shape[0]-2)  ]),
                       r = np.concatenate([ np.array([-1., 1]), np.zeros(M1.shape[0]-2) ] )
                     ) 
        return M1, M2, D
    
    def fit(self, mu, policy_factor, test_val = None,  **kwargs):
        
        P_e_given_i = self._make_P_e_given_i(self.init_kwargs["P_e_given_i"], self.days_total)
        M1, M2, D = self._make_pior_mats(self.T_period, self.days_total)
        if test_val is None:
            test_val = np.ones(len(self.event_series))/len(self.event_series)
        
        p_i_samples, self.posterior = self._sample_posterior( counts_arr = self.event_series.values.copy() , 
                                                              P_e_given_i = P_e_given_i, 
                                                              M1 = M1,
                                                              M2 = M2, 
                                                              D = D, 
                                                              mu = mu, 
                                                              days_total = self.days_total, 
                                                              test_val = test_val,
                                                              policy_dates = self.policy_dates, 
                                                              policy_factor = policy_factor, 
                                                             **kwargs)
        self.p_i_samples = pd.DataFrame( index = self.event_series.index ,
                                         data =  p_i_samples.transpose() )
        self.p_i = self._samples_to_cred_interval( self.p_i_samples,  sample_axis =1  )
        
        ## caculate R_e
        self.Re_samples = self._est_Re(self.p_i_samples, self.T_period )
        self.Re = self._samples_to_cred_interval(self.Re_samples ,  sample_axis =1  )
    
        return  self.p_i.copy(), self.Re.copy(), self.p_i_samples.copy(), self.Re_samples.copy() 
                                 
    @staticmethod
    def _sample_posterior(counts_arr, P_e_given_i, M1, M2, D, mu, days_total, test_val,
                          policy_dates, policy_factor, **kwargs):
        
        n = int(counts_arr.sum())
        ## multiply mu by policy_factor to allow more change on policy_dates
        mu_hard = mu*np.ones(D.shape[0])
        mu_soft = mu_hard.copy()
        policy_day_shift =  np.argmax( M1[0,:]) + 1
        for start, stop in policy_dates:
            mu_soft[ start-policy_day_shift :stop-policy_day_shift] *= policy_factor
        
        with pm.Model() as model:
            p_i = Infect_Series_Prior("p_i",
                                       mu_soft=mu_soft ,
                                       mu_hard = mu_hard, 
                                       M1=M1, 
                                       M2 =M2, 
                                       D= D, 
                                       G = P_e_given_i,
                                       N =n,
                                       testval = test_val, 
                                       shape= days_total,
                                       transform=pm.distributions.transforms.stick_breaking)
            p_e = tt.dot(P_e_given_i, p_i ) 
            p_e_given_observed =  p_e/ tt.sum(p_e)
            N = pm.Multinomial("N", n=n, p=p_e_given_observed, observed= counts_arr)
               
            posterior = pm.sample(cores= 1, **kwargs) 
            
        return posterior["p_i"].copy(), posterior
    
    @staticmethod
    def _est_Re(p_i_samples, T_period,  min_infected_frac= 0.0001 ):
        """
        p_i_samples  - DataFrame with 
                                values — proportional to number of infected people 
                                index — dates, 
                                columns — posterior samples
        T_period — 2-tuple
        min_infected_frac — float                 
        trans_period  - 2-tuple representing left end open right end closed interval
        """
        Re_idx_min =  max(T_period[1]-1, np.argmax(p_i_samples.mean(axis = 1).values > min_infected_frac) )
        Re_samples = pd.DataFrame(index = p_i_samples.index[Re_idx_min:].copy(), 
                                 columns = p_i_samples.columns.copy(),
                                 data = 0. )
        for idx in range(Re_idx_min, p_i_samples.shape[0]):
            Re_samples.iloc[idx - Re_idx_min, :] = (p_i_samples.iloc[idx, :]
                                                    )/( (p_i_samples.iloc[idx - T_period[1]+1 : idx - T_period[0] + 1, :]
                                                                ).sum(axis = 0) 
                                                       )
        Re_samples= Re_samples*(T_period[1] - T_period[0])
        return Re_samples
    
    
    def predict_p_e(self,):
        
        P_e_given_i = Infection_Series_Estimator._make_P_e_given_i(self.init_kwargs["P_e_given_i"], self.days_total, observed_only = False )
        p_e_samples = P_e_given_i@(self.p_i_samples.values)
        
        p_e_samples = pd.DataFrame(data = p_e_samples, columns = self.p_i_samples.columns)
        p_e_index_observed = self.p_i_samples.index.copy()
        p_e_index_unobserved = p_e_index_observed[-1] + pd.timedelta_range( start = pd.Timedelta(days = 1), 
                                                                             end =   pd.Timedelta(days = len(p_e_samples) - len(p_e_index_observed) ),
                                                                              freq = 'd' )
        p_e_index = p_e_index_observed.append(p_e_index_unobserved)          
        p_e_samples.index= p_e_index 
        p_e = self._samples_to_cred_interval(p_e_samples ,  sample_axis =1  )
        
        ## Expected number of events per day
        n_observed = self.event_series.sum()
        n_e_samples = p_e_samples.apply(
                            lambda x: x*n_observed*np.concatenate([ np.ones(len(p_e_index_observed))/x.loc[p_e_index_observed].sum(),
                                                                    (1./x.loc[p_e_index_observed].sum()
                                                                        )*np.ones(len(p_e_index_unobserved))
                                                              ] ),
                            axis = 0)
        n_e = self._samples_to_cred_interval(n_e_samples, sample_axis =1  )
                 
        return p_e,  p_e_samples, n_e, n_e_samples
    
    @staticmethod
    def _samples_to_cred_interval(samples , sample_axis =1 ):

        if isinstance(samples, pd.Series):
            out = pd.Series( [ samples.mean(), samples.quantile(0.025), samples.quantile(0.975)]  , index = ["mean" , "lower" , "upper"] )
        elif isinstance(samples, pd.DataFrame ):
            out = pd.concat([ samples.mean(axis = sample_axis),
                            samples.quantile(0.025, axis = sample_axis), 
                            samples.quantile(0.975, axis = sample_axis)
                            ],
                            axis = sample_axis, keys = ["mean" , "lower", "upper"] )
        else:
            raise ValueError()
        return out
    
class Infect_Series_Prior( pm.Continuous):
    def __init__(self, mu_soft, mu_hard, M1, M2, D, G, N, *args, **kwargs):
        super(Infect_Series_Prior, self).__init__(*args, **kwargs)
        
        self.mu_soft = mu_soft
        self.mu_hard = mu_hard
        self.M1 = M1
        self.M2 = M2
        self.D = D
        self.G = G
        self.N = N
        ## define mask used in jacobian for transmission prior
        
    def logp(self, value):
        mu_soft = self.mu_soft
        mu_hard = self.mu_hard
        M1 = self.M1
        M2 = self.M2
        D = self.D
        G = self.G
        N = self.N
        
        diff = tt.dot(D, tt.log(tt.dot(M1,value)) - tt.log(tt.dot(M2,value)))
        t_out = -1.*tt.sum( tt.switch(tt.lt(diff, 0.), mu_soft, mu_hard )*diff**2  )             
        return t_out
    
class Policy_Stats(object):
    """Class for relating changes in policy to changes in statictics of samples from the posterior
    distribution that are calcuatled over time intervals before and after the change. 
    """
    def __init__(self, policy_dates, max_before =7, max_after = 7):
        """
        policies — DataFrame - index : policy names
                               columns : states/provinces
        """
        self.max_before = max_before
        self.max_after = max_after
        self._policy_dates = Policy_Stats._preprocess_policy_dates(policy_dates)
        
        self._compare_intervals = self._policy_dates.groupby(level = 0).apply(
                                                                lambda x: Policy_Stats._make_compare_intervals(x[x.name]) 
                                                                        )
        self.change_samples = None
        self.change_interval = None
    @property
    def policy_dates(self):
        """
        pd.Series Multiindex with level0 — state and level1 — policy name 
                  Values are dates
        """
        return self._policy_dates.copy()
    @property
    def compare_intervals(self):
        """
        pd.DataFrame: index - level 0 - state; level 1 - date
                      columns before_start, before_end, after_start, after_end, policy
                      values - TimeStamps or string
        """
        return self._compare_intervals.copy()
    
    @staticmethod  
    def _preprocess_policy_dates(policy_dates):
        processed = policy_dates.applymap(lambda x: pd.Timestamp(x)
                                         ).dropna(axis =1, how = "all")
        processed = processed.transpose().stack()
        return  processed 
    
    @staticmethod  
    def _make_compare_intervals( p_series, max_before =7, max_after = 7, truncate_before = False ):
        """
        Construct time-intervals before and after implementation of a policy, considering the dates at
        which other policies are implemented.
        Inputs
        ------
            p_series — pd.Series: index - policies,
                                 values - dates implemented
            max_before - int: the number of days before implementation of policy to include
            max_after - int: the number of days after implementation policy (including the day implementation) to include
            truncate_before — If the "before" interval overlaps the implementation of a previous policy do we trucate 
                            to only the later part of the interval? 
        Returns
        -------
            compare_intervals- index - level 0 - state; level 1 - date
                                  columns before_start, before_end, after_start, after_end, policy
                                values - TimeStamps or string
        """
        
        p_series = p_series.squeeze().sort_values().dropna()
        dates_unique = [pd.Timestamp(x) for x in sorted(p_series.unique())]

        date_to_p = OrderedDict([])
        for i, d in enumerate(dates_unique):
            if not date_to_p:  ## the first date
                before_start = d + timedelta(days=-max_before) 
                before_end = d 
                after_start = d
                if i + 1 < len(dates_unique): ## there is another entry
                    after_end = min( dates_unique[i+1], d + timedelta(days=max_after) )
                else:
                    after_end = d + timedelta(days=max_after) 
            else:
                if truncate_before:
                    before_start = max( d+timedelta(days=-max_before), dates_unique[i-1] + timedelta(days=1) )
                else:
                    before_start = d + timedelta(days=-max_before)
                before_end = d 
                after_start = d
                if i + 1 < len(dates_unique): ## there is another entry
                    after_end = min( dates_unique[i+1], d + timedelta(days=max_after) )
                else:
                    after_end = d + timedelta(days=max_after) 
            date_to_p[d] = [ before_start, before_end, after_start, after_end ] 

        compare_intervals = pd.DataFrame.from_dict( date_to_p,
                                                   orient = "index", 
                                                   columns = [ "before_start", "before_end", "after_start", "after_end"])
        ## Add policies for each date
        compare_intervals["policy"] = ""
        for p, d in p_series.iteritems():
            if compare_intervals.loc[d, "policy"]:
                compare_intervals.loc[d, "policy"] = compare_intervals.loc[d, "policy"] + "_AND_" + p
            else:
                compare_intervals.loc[d, "policy"] = p            
        return compare_intervals
    
    def est_pct_change(self, samples_dict, policies):
        """
        Inputs
        ------
            samples_dict — dictionary: keys -states
                                       values - dataframe with dates as index and samples as columns
            policies -  None or list of strings appearing in self.policy_dates index, level1
        Returns:
            pct_change_interval — pd.DataFrame: index: level0 state; level1 policy,
                                           columns: mean, lower, upper
                                           values: percent change
            pct_samples — pd.DataFrame: index: level0 state; level1 policy,
                                           columns: samples index
                                           values: percent change
        """
        c_intervals = self.compare_intervals.copy()
        change_func =  lambda before, after: (after.mean(axis = 0) - before.mean(axis = 0)).divide(before.mean(axis = 0))*100.
        self.change_samples  = self._est_change(samples_dict, c_intervals, change_func = change_func,  policies = policies)
        ## Get credible interval
        self.change_interval= self._samples_to_cred_interval( self.change_samples , sample_axis = 1)
        return self.change_interval.copy(), self.change_samples.copy()

    @staticmethod
    def _est_change(samples_dict, c_intervals, change_func,  policies = None):
        """
        
        Inputs
        ------
            samples_dict — dictionary: keys -states
                                       values - dataframe with dates as index and samples as columns
            c_intervals - same as self.compare_intervals
            change_func - callable: arguments: before (pd.DataFrame), after (pd.DataFrame)
                                            where each columns is a sample and index is dates from before or after interval
                                    returns: series where index is samples and values are the change statisitic for the sample  
            policies -  None or list of strings appearing in self.policy_dates index, level1
        Returns
        -------
            change_samples — pd.DataFrame: index: level0 state; level1 policy,
                                           columns: samples index
                                           values: change statisitc
        """
        samples_dict= {s: df for s, df in samples_dict.items() if s in c_intervals.index.levels[0]} ## filter to states with policy data
        results_dict = {}
        for state, samples in samples_dict.items():
            ## Get dates and names of policies
            if policies is None:
                dates_interest =  list( c_intervals.loc[state,:].index )
            else:
                dates_interest = [d for d, row in c_intervals.loc[state,: ].iterrows() 
                                      if set(policies).intersection(row["policy"].split("_AND_"))  ]
            policies_interest =  list(c_intervals.loc[ [(state, d) for d in dates_interest] , "policy"])
            changes_by_sample = pd.DataFrame( index =  policies_interest, columns = samples.columns, data = 0. )
            ### Compute change at each policy/date
            for d, p in zip(dates_interest, policies_interest):
                before_start, before_end, after_start, after_end = c_intervals.loc[(state, d), 
                                                                         ["before_start", "before_end", "after_start", "after_end"]]
                changes_by_sample.loc[p, :] = change_func( samples.loc[ before_start:before_end , :],  
                                                    samples.loc[ after_start: after_end , :]).values
            ## Update with results for state
            results_dict[state] = changes_by_sample
        ## Combine to single dataframe
        keys_tmp = list(results_dict.keys())
        change_samples = pd.concat( [results_dict[k] for k in keys_tmp] , axis = 0,  keys = keys_tmp)
        ## Add policy_date to index
        change_samples["policy_date"] = pd.NaT
        for (state,date), row in c_intervals.iterrows():
             change_samples.loc[ (state, row["policy"]), "policy_date"] = date
        change_samples = change_samples.set_index( "policy_date" , append = True)
        change_samples.index.names = ["state/province" , "policy" , "policy_date"]
        return change_samples 

    def boxplot_changes_by_state(self, states = None, policies = None, change_samples =None ,
                                aspect= 3, height = 6, hspace = 1.):
        ####### Process inputs
        if change_samples is None:
            if self.change_samples is None:
                raise Exception("change_samples not set eiter provide as method argument or run est_pct_change")
            else:
                change_samples = self.change_samples
        ## states
        if states is None:
            states = list(change_samples.index.levels[0])
        else:
            assert all([s in change_samples.index.levels[0] for s in states]),"No data for some states in provided list" 
            change_samples = change_samples.loc[states, : ].copy()
        if policies is not None:
            mask = [ len(set(i[1].split("_AND_")).intersection(policies)) > 0 for i in change_samples.index ]
            change_samples_mask = change_samples_mask.loc[mask, :].copy()
        ####### Helper functions
        def format_xlab(s, width = 12):
            s = "\nAND\n".join([textwrap.fill(x, width = width) for x in s.split("_AND_")])
            return s
        def plot_func(data, y, color):
            policies_and_dates = data[["policy", "policy_date"]].drop_duplicates().sort_values(by = "policy_date").reset_index(drop =True)
            ax = sns.boxplot(data = data, x = "policy" , y = y, order = list(policies_and_dates["policy"].values),  whis=[5, 95])
            ax.set_xticklabels([format_xlab(t, width = 20) for t in policies_and_dates["policy"].values] , rotation = 90)
            return None
        ####### PLOT
        change_samples = change_samples.transpose().melt()
        g = sns.FacetGrid(change_samples, row="state/province", aspect= aspect, height = height,sharex=False, )
        g.map_dataframe(plot_func, y = "value")
        for ax in np.ravel(g.axes):
            ax.set_ylabel("Percent change in " + r'$R_e$')
            ax.set_ylim(None, 50)
            ax.grid(True)
        g.fig.subplots_adjust(hspace=hspace)
        return g.fig
    
    def boxplot_changes_by_policy(self, states = None, policies = None,  change_samples = None,
                                  y_label = "Percent change in " + r'$R_e$', aspect = 4, height = 6,y_lim = (-100, 100),
                                 hspace = 3.2 ):
        ####### Process inputs
        if change_samples is None:
            if self.change_samples is None:
                raise Exception("change_samples not set eiter provide as method argument or run est_pct_change")
            else:
                change_samples = self.change_samples
        if states is None:
            states = list(change_samples.index.levels[0])
        else:
            assert all([s in change_samples.index.levels[0] for s in states]),"No data for some states in provided list" 
            change_samples = change_samples.loc[states, : ].copy()
        if policies is None:
            policies = list(np.unique( [ x for y in change_samples.index.levels[1] for x in y.split("_AND_")] ) )
        ####### Helper functions
        def format_xlab(s, width = 30):
            s = "\nAND\n".join([textwrap.fill(x, width = width, initial_indent = '   ', subsequent_indent= '   '
                                              ) if i>0 else textwrap.fill(x, width = width
                                                                             ) for i,x in enumerate(s.split("_AND_")) ])
            return  s
        ####### PLOT
        nrows = len(policies)
        fig, axes = plt.subplots(figsize = (height*aspect, height*nrows), nrows = nrows , ncols = 1 )
        axes = np.ravel(axes)
        for p, ax in zip(policies, axes):
            mask = [ len(set(i[1].split("_AND_")).intersection([p])) > 0 for i in change_samples.index ]
            plot_data = change_samples.loc[mask, :].reset_index()
            plot_data["state/province_and_policy"] =  plot_data.apply(lambda x: x["state/province"] +" : " + x["policy"], axis = 1)
            plot_data = plot_data.drop(columns = ["state/province" , "policy" , "policy_date"])
            plot_data = plot_data.set_index("state/province_and_policy", verify_integrity=True)
            plot_data = plot_data.transpose().melt()
            ax = sns.boxplot(data = plot_data , x =  "state/province_and_policy" , y = "value", ax= ax, whis=[5, 95])
            ax.set_xticklabels( [format_xlab(x.get_text()) for x in  ax.get_xticklabels()], rotation = 90)
            ax.set_ylim( y_lim )
            ax.set_ylabel(y_label)
            ax.grid(True)
            ax.set_title(p)
        fig.subplots_adjust(hspace = hspace)
        return fig
    
    @staticmethod
    def _samples_to_cred_interval(samples, sample_axis = 1):
        """
        Construct a symmetric 95% credible interval (CI) (TODO: HDI  interval)
        Inputs
        ------
            samples — pd.DataFrame or pd.Series where one axis stores samples that are to be aggregated to CI
            samples_axis- int 
        """
        if isinstance(samples, pd.Series):
            out = pd.Series( [ samples.mean(), samples.quantile(0.025), samples.quantile(0.975)]  , index = ["mean" , "lower" , "upper"] )
        elif isinstance(samples, pd.DataFrame ):
            out = pd.concat([ samples.mean(axis = sample_axis),
                            samples.quantile(0.025, axis = sample_axis), 
                            samples.quantile(0.975, axis = sample_axis)
                            ],
                            axis = sample_axis, keys = ["mean" , "lower", "upper"] )
        else:
            raise ValueError()
        return out

plt.plot(make_infect_to_death_pmf())
_=plt.xlabel("days")
_= plt.ylabel("g(s-t)")
########################################
###        GLOBALS               #######
########################################

policies_string= StringIO("""
Policy,New York,New Jersey,Michigan,Louisiana,Massachusetts,Illinois,Connecticut,California,Pennsylvania,Florida,Georgia,Washington
Emergency Declaration,3/7/20,3/9/20,3/10/20,3/11/20,3/10/20,3/9/20,3/10/20,3/4/20,3/6/20,,,
Bar/Restaurant Limits,3/16/20,3/16/20,3/16/20,3/17/20,3/15/20,3/17/20,3/17/20,3/16/20,3/16/20,,,
School Closure,3/18/20,3/18/20,3/16/20,3/16/20,3/15/20,3/17/20,3/17/20,3/15/20,3/13/20,,,
Stay At Home Order,3/20/20,3/21/20,3/24/20,3/22/20,3/24/20,3/20/20,3/23/20,3/19/20,4/1/20,,,
Non-essential Business Closures,3/22/20,3/21/20,3/24/20,3/22/20,3/24/20,3/20/20,3/24/20,3/19/20,3/22/20,,,
Gatherings Ban 500 Or Stricter,3/12/20,,,,,,3/12/20,,,,,
Gatherings Ban 50 Or Stricter,3/16/20,3/16/20,3/17/20,3/17/20,3/15/20,3/18/20,3/16/20,,,,,
Gatherings Ban 10 Or Any Size,3/20/20,3/21/20,3/24/20,3/22/20,3/23/20,3/20/20,3/26/20,3/16/20,,,,
""")
policies = pd.read_csv(policies_string, index_col = 0)
JHU_deaths = load_JHU_deaths()
## MODEL PARAMETERS
T_period = (1,8)
draws =  500
chains = 4
mu = 13.
policy_factor = 0.01

print("GLOBALS")
print("\t Policy dates (manually currated)")
display(policies)
print("\t JHU covid death data (aggregated by US state)")
display(JHU_deaths.head())
########################################################################
#### SHOW CUMULATIVE DEADTH FOR 10 US STATES WITH HIGHEST TOTAL DEATHS ###
########################################################################
state_top_cas = list(JHU_deaths.iloc[:,-1].sort_values()[::-1].index[0:10])

ax = JHU_deaths.loc[state_top_cas, : ].transpose().plot()
leg = ax.get_legend()
leg.set_bbox_to_anchor( (1, 1), )
#######################################################################################
###    SMOOTHONG                                                                        ###
#### ALTHOUGH IT APPEARS THAT THE JHU DATA STORES CUMULATIVE DEATHS (ABOVE PLOT)        ###
#### THERE ARE A FEW INSTANCES WHERE THE CUMULATVE DEATHS ARE HAVE NAVE NEGATIVE SLOPE  ###
#### WE HANDLE THIS BY AVERAING WITH A 7 DAY WINDO AROUND CASES WHERE DEATHS PER DAY    ###
#### GO NEGATIVE. THE NUMBER OF DAYS AT WHICH THIS MUST BE DONE IS SMALL AND IT ONLY   ### 
#### NEEDS TO BE DONE FOR NY AND CA                                                     ###
########################################################################
deaths_by_day = pd.DataFrame( index = pd.to_datetime(JHU_deaths.columns[1:]), columns = state_top_cas)
is_smoothed = pd.Series(index =state_top_cas, data = False, dtype = bool )

for state in state_top_cas:
    deaths_by_day_state =  JHU_deaths.loc[state,:].diff().dropna()
    deaths_by_day_state_smooth, smoothed = simple_smoother( deaths_by_day_state.copy(), window_size = 7)
    deaths_by_day.loc[:, state] = deaths_by_day_state_smooth.values
    is_smoothed.loc[state] = smoothed
display(is_smoothed)
################################################################################
###  PLOTING NUMBER OF DEATHS PER DAY  — Just to get a feel for data        ###
################################################################################

fig , ax = plt.subplots(figsize= (16,6))
ax = deaths_by_day.divide(deaths_by_day.max(axis = 0), axis = 1).rolling(3,).mean().loc['2020-03-05':].plot(ax = ax)
leg = ax.get_legend()
leg.set_bbox_to_anchor( (1, 1), )
_ =ax.set_ylabel("Deaths by day\n(scaled by maximum)")
######################################
#### MODEL FITTING         ###########
#####################################


## Set up directories for storing ouput
time = datetime.datetime.isoformat(datetime.datetime.now() ).split(".")[0]
models_dir = os.path.join( "models", str(time) )
preds_dir = os.path.join( "predictions", str(time) )

os.makedirs(models_dir , exist_ok = True)
os.makedirs(preds_dir, exist_ok = True)

## Do estimation and write results
for state in state_top_cas:
    print("Working on {}".format(state))
    policy_dates =   [x for x in policies[state].dropna().unique() ]
    ise = Infection_Series_Estimator(event_series =deaths_by_day[state].copy().astype(int), 
                                     P_e_given_i= make_infect_to_death_pmf(), 
                                     T_period = T_period,
                                     policy_dates= policy_dates )
    p_i, Re, p_i_samples, Re_samples= ise.fit(mu = mu,
                                              policy_factor = policy_factor, 
                                              draws = draws ,
                                              chains = chains )
    
    df_samples = pd.concat( [p_i_samples ,Re_samples], axis = 1, keys = ["p_i", "Re"] )
    df_samples.to_csv( os.path.join(preds_dir , "{}_samples.csv".format(state)) )
    df_cinterval =  pd.concat( [p_i , Re] , axis = 1, keys = ["p_i", "Re"] )
    df_cinterval.to_csv( os.path.join(preds_dir , "{}_intervals.csv".format(state)) )
    
    f = open(os.path.join(models_dir, "{}.pkl".format(state)) , 'wb' )
    pickle.dump(ise , f)
    f.close()
    
##  Combine results
df_samples_list = [ pd.read_csv( os.path.join(preds_dir,"{}_samples.csv".format(state)), index_col=0 , header = [0,1] ) 
                                    for state in list(state_top_cas )  ]
df_samples = pd.concat(df_samples_list, axis = 1, keys = list(state_top_cas ) )
df_samples.index = pd.to_datetime(df_samples.index)
df_samples.to_csv( os.path.join(preds_dir,"samples.csv" ) )
for state in state_top_cas:
    os.remove(os.path.join(preds_dir,"{}_samples.csv".format(state)) )
    
df_cintervals_list = [ pd.read_csv( os.path.join(preds_dir,"{}_intervals.csv".format(state)), index_col=0 , header = [0,1] ) 
                                    for state in list(state_top_cas )  ]
df_cintervals = pd.concat(df_cintervals_list, axis = 1, keys = list(state_top_cas) )
df_cintervals.to_csv(os.path.join(preds_dir,"intervals.csv" ) )
df_cintervals.index = pd.to_datetime(df_cintervals.index)
for state in state_top_cas:
    os.remove(os.path.join(preds_dir,"{}_intervals.csv".format(state)) )
ncols = 2
models_dir = models_dir
nrows = -(-len(state_top_cas)//2)


fig , axes = plt.subplots(nrows = nrows, ncols = ncols ,figsize = (16,3*nrows))
axes = np.ravel(axes)

for state, ax in zip(state_top_cas, axes) :
    f = open( os.path.join(models_dir, "{}.pkl".format(state) ) , 'rb')
    ise = pickle.load(f)
    f.close()
    (ise.predict_p_e()[2]).plot(ax = ax)
    deaths_by_day[state].plot(marker = "o", ax = ax)
    ax.set_title(state)
    
fig.tight_layout()
## Load data
preditions_dir = preds_dir ## Change this to match the folder created during execution of estimation cell 
df_cintervals = pd.read_csv(os.path.join(preditions_dir, "intervals.csv"), header = [0,1,2], index_col = 0)
fig = plot_shaded(df_cintervals.loc[ '2020-02-25': ,(slice(None), "Re") ].droplevel(axis = 1,level =1), alpha = 0.1, figsize=(18,6))
ax = fig.get_axes()[0]
ax.set_ylim((0 , 7) )
ax.grid(True)
leg = ax.get_legend()
leg.set_bbox_to_anchor((1,1))
_ =ax.set_ylabel(r'$R_e$', fontsize = 'xx-large')
fig = plot_shaded(df_cintervals.loc[ '2020-03-05': ,(slice(None), "p_i") ].droplevel(axis = 1,level =1),
                                           alpha = 0.1,figsize = (16,6))
ax = fig.get_axes()[0]
ax.set_ylim((0 , 0.15) )
ax.grid(True)
leg = ax.get_legend()
leg.set_bbox_to_anchor((1,1))
_=ax.set_ylabel('Fraction infected on each day', fontsize = 'xx-large')
## File names
preditions_dir = preds_dir 

## Load Data
df_samples = pd.read_csv( os.path.join(preditions_dir, "samples.csv"), index_col = 0, header=[0,1,2] )
df_samples.index = [pd.Timestamp(x) for x in df_samples.index]
re_samples = df_samples.loc[ : , (slice(None), "Re")].copy().droplevel(axis=1,  level = 1)
re_samples_grouped = re_samples.groupby(axis = 1, level = 0 )

## Create policy_stats object
policy_stats = Policy_Stats(policies)
pct_change_interval, pct_change_samples = policy_stats.est_pct_change( 
                                                {state:  df.droplevel(axis =1 , level = 0) for state, df in re_samples_grouped}, 
                                                policies =None)
## Plot changes in Ro grouped by state
fig = policy_stats.boxplot_changes_by_state(policies =None, aspect = 5, height = 4, hspace=1.7, )
## Plot changes in Ro groups by policy
fig = policy_stats.boxplot_changes_by_policy(aspect = 4, height = 6,y_lim = (-100, 100), hspace = 3.2)

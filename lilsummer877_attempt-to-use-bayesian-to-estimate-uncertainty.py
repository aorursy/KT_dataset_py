import os

import pandas as pd

import matplotlib.pyplot as plt



from  datetime import datetime, timedelta

import gc

import numpy as np

#plt.style.use('ggplot')
%matplotlib inline

import sys

import re



plt.style.use('seaborn-darkgrid')

import seaborn as sns

import patsy as pt

import pymc3 as pm



plt.rcParams['figure.figsize'] = 14, 6

np.random.seed(0)

print('Running on PyMC3 v{}'.format(pm.__version__))
#os.listdir('../m5-forecasting-uncertainty/')
submission = pd.read_csv('../input/m5-forecasting-uncertainty/sample_submission.csv')
#submission.shape
#submission.head()
sale = pd.read_csv('../input/m5-forecasting-uncertainty/sales_train_validation.csv')
#sale.head()
sale.shape
total_historical = sale.iloc[:,6:].sum()
total_historical.shape
calendar = pd.read_csv('../input/m5-forecasting-uncertainty/calendar.csv')
calendar['event_true_1'] = calendar.event_name_1.notna()

calendar['event_true_2'] = calendar.event_name_2.notna()



calendar['event_true_all'] = calendar.event_true_1 + calendar.event_true_2

calendar['event_true_all'] = calendar.event_true_all.apply(lambda x: x>0)

calendar['event_true_all'] = calendar.event_true_all.astype('int')

calendar['date'] = pd.to_datetime(calendar.date)
#calendar.dtypes
#calendar.columns
calendar['d_parse'] = calendar.d.apply(lambda x: int(x.split('_')[1]))
#calendar.head()
calendar_feature = calendar[['wm_yr_wk', 'wday', 'month', 'year', \

       'snap_CA', 'snap_TX', 'snap_WI', \

       'event_true_all', 'd_parse']]
calendar_feature.dtypes
# specify formula

fml = 'total ~ wday + month + year + snap_CA + snap_TX + snap_WI + event_true_all + d_parse'
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

scaler = StandardScaler()

#minmax = MinMaxScaler()
calendar_feature = calendar[['wm_yr_wk', 'wday', 'month', 'year', \

       'snap_CA', 'snap_TX', 'snap_WI', \

       'event_true_all', 'd_parse']]



scaled_feature = pd.DataFrame(scaler.fit_transform(calendar_feature))

scaled_feature.columns = calendar_feature.columns

scaled_feature.min()
np.where(total_historical < 10000)[0]
total_historical.iloc[[ 330,  696, 1061, 1426, 1791]]=np.quantile(total_historical, 0.025)
np.min(total_historical)
#minmax_feature.iloc[:1913,9]
# create data frame

df = scaled_feature.iloc[:1913,:]

df.loc[:,'total'] = total_historical.values

df.loc[:, 'd_parse'] = calendar_feature.iloc[:1913, 8] - np.min(calendar_feature.d_parse) + 1

df.head()
(mx_en, mx_ex) = pt.dmatrices(fml, df, return_type='dataframe', NA_action='raise')

pd.concat((mx_ex.head(3),mx_ex.tail(3)))

with pm.Model() as mdl_first:



    # define priors, weakly informative Normal

    # here we tried to remove all the time variable and 

    # treat all these as 'attributes' of data rather than the exposure

    b0 = pm.Normal('b0_intercept', mu=0, sigma=1)

    b2 = pm.Normal('b2_wday', mu=0, sigma=1)

    b3 = pm.Normal('b3_month', mu=0, sigma=1)

    b4 = pm.Normal('b4_year', mu=0, sigma=1)

    b5 = pm.Normal('b5_snapCA', mu=0, sigma=1)

    b6 = pm.Normal('b6_snapTX', mu=0, sigma=1)

    b7 = pm.Normal('b7_snapWI', mu=0, sigma=1)

    b8 = pm.Normal('b8_event_true_all', mu=-0.01, sigma=1)



    # define linear model and exp link function

    theta = (b0 +

            b2 * mx_ex['wday'] +

            b3 * mx_ex['month'] + 

            b4 * mx_ex['year'] + 

            b5 * mx_ex['snap_CA'] + 

             b6 * mx_ex['snap_TX'] + 

             b7 * mx_ex['snap_WI'] + 

             b8 * mx_ex['event_true_all'] + 

              np.log(mx_ex['d_parse'] ))  ## there is the log(t) as an offset



    ## Define Poisson likelihood

    y = pm.Poisson('y', mu=np.exp(theta), observed=mx_en['total'].values)
with mdl_first:

    trace = pm.sample(1000, tune=2000, init='adapt_diag', target_accept =.8)
mdl_first.check_test_point()
## helper function from pymc documentation

def strip_derived_rvs(rvs):

    '''Convenience fn: remove PyMC3-generated RVs from a list'''

    ret_rvs = []

    for rv in rvs:

        if not (re.search('_log',rv.name) or re.search('_interval',rv.name)):

            ret_rvs.append(rv)

    return ret_rvs





def plot_traces_pymc(trcs, varnames=None):

    ''' Convenience fn: plot traces with overlaid means and values '''



    nrows = len(trcs.varnames)

    if varnames is not None:

        nrows = len(varnames)



    ax = pm.traceplot(trcs, var_names=varnames, figsize=(12,nrows*1.4),

                      lines=tuple([(k, {}, v['mean'])

                                   for k, v in pm.summary(trcs, varnames=varnames).iterrows()]))



    for i, mn in enumerate(pm.summary(trcs, varnames=varnames)['mean']):

        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data',

                         xytext=(5,10), textcoords='offset points', rotation=90,

                         va='bottom', fontsize='large', color='#AA0022')
rvs_fish = [rv.name for rv in strip_derived_rvs(mdl_first.unobserved_RVs)]

pm.summary(trace, varnames=rvs_fish)
pm.plot_trace(trace)
with mdl_first:

    pp_trace = pm.sample_posterior_predictive(trace, var_names=rvs_fish, samples=4000)
df_2 = scaled_feature.iloc[1913:,:]

total_id = [i for i in submission.id if 'Total' in i]

# change back d_parse

df_2['d_parse']= calendar_feature.iloc[1913:,:].d_parse.values

df_2.d_parse.max()
submission_validation = df_2.iloc[:28, :]

submission_evaluation = df_2.iloc[28:, :]

submission_validation.shape,submission_evaluation.shape
pp_trace.keys()
pp_trace['b0_intercept']
def return_y(df):

    result = 1*pp_trace['b0_intercept']

    for (i,j) in zip([*pp_trace.keys()][1:], df.index[1:]):

        #print(i, j)

        result += pp_trace[i]*df[j]

        #print(result)

    return np.exp(result + np.log(df['d_parse']))

    #return result

validation_y = np.zeros((28, 4000))

evaluation_y = np.zeros((28, 4000))
submission_validation.iloc[0].index
#submission_evaluation.iloc[0]
for row in range(len(submission_validation)):

    validation_y[row, :] = return_y(submission_validation.iloc[row])

    evaluation_y[row, :] = return_y(submission_evaluation.iloc[row])
np.mean(validation_y)
np.mean(total_historical)
## organize the data

total_qt = [float(i.split('_')[2]) for i in total_id]



total_only_submission = submission[submission.id.isin(total_id)]



total_only_submission['qt']=total_qt



total_only_submission.reset_index(inplace=True)



total_only_submission.loc[:7]
for i in range(1,29):

    col_name = 'F' + str(i)

    total_only_submission.loc[:8,col_name] =np.quantile(validation_y[i-1], total_qt[:9])



for i in range(1,29):

    col_name = 'F' + str(i)

    total_only_submission.loc[9:,col_name] =np.quantile(evaluation_y[i-1], total_qt[:9])
total_only_submission
total_only_submission.to_csv('total_submission.csv', index=False)
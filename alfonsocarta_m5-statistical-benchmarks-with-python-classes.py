from tqdm import tqdm

import matplotlib.pyplot as plt

%matplotlib inline



import os

import seaborn as sns

import pandas as pd

import numpy as np

from itertools import cycle

from scipy.stats import hmean



import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as pyo

from scipy.optimize import minimize_scalar

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing

import itertools

from functools import partial

from multiprocessing import Pool

import statsmodels.api as sm

import warnings

from statsmodels.tsa.api import SimpleExpSmoothing

from scipy.ndimage.interpolation import shift



pyo.init_notebook_mode(connected=True)

import math



from typing import Union

from tqdm.auto import tqdm as tqdm



import m5_constants as cnt



pd.set_option('max_columns', 50)
def reduce_mem_usage(df, verbose=True):

    """

    from M5 Forecast: Keras with Categorical Embeddings V2

    https://www.kaggle.com/mayer79/m5-forecast-keras-with-categorical-embeddings-v2

    """

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'\

                      .format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df







def load_raw_data():

    """

    Load raw input data. Paths are in the constant.py file

    

    Return: 

     - df_train_val : sales train val data-frame

     - df_calendar : calendar data-frame

     - df_price : price data-frame

     - df_sample_sub : sample submission data-frame

    """

    df_train_val = pd.read_csv(cnt.SALES_TRAIN_VAL_PATH)

    df_calendar = pd.read_csv(cnt.CALENDAR_PATH)

    df_price = pd.read_csv(cnt.SELL_PRICE_PATH)

    df_sample_sub = pd.read_csv(cnt.SAMPLE_SUBMISSION)

    

    df_train_val = reduce_mem_usage(df_train_val)

    df_calendar = reduce_mem_usage(df_calendar)

    df_price = reduce_mem_usage(df_price)

    df_sample_sub = reduce_mem_usage(df_sample_sub)

    

    print("df_train_val shape: ", df_train_val.shape)

    print("df_calendar shape: ", df_calendar.shape)

    print("df_price shape: ", df_price.shape)

    print("df_sample_sub shape: ", df_sample_sub.shape)

    

    return df_train_val, df_calendar, df_price, df_sample_sub



def split_train_val_sales(df_sales, horizon):

    """

    train-val split of sales data according to the horizon parameter

    """

    

    df_sales_train = df_sales.iloc[:,:-horizon]

    df_val_item = df_sales[['id','item_id','dept_id','cat_id','store_id','state_id']]

    df_val_qty = df_sales.iloc[:,-cnt.HORIZON:]

    df_sales_val = pd.concat([df_val_item, df_val_qty], axis=1)

    

    print("df_sales_train shape: ", df_sales_train.shape)

    print("df_sales_val shape: ", df_sales_val.shape)

    

    return df_sales_train, df_sales_val
# Load Raw input data

df_train_val, df_calendar, df_price, df_sample_sub = load_raw_data()
%%time

# Split train val sales

df_train, df_val = split_train_val_sales(df_sales=df_train_val, horizon=cnt.HORIZON)
df_train.head()
df_val.head()
df_calendar.head()
df_price.head()
df_sample_sub.head()
df_sample_sub.tail()
def plot_time_series(index, df_train, calendar, df_eval=None, preds=None):

    

    df_eval = df_val.copy()

    id_columns = [i for i in df_train_val.columns if not i.startswith('d_')]

    d_columns_train =  [i for i in df_train.columns if i.startswith('d_')]

    if df_eval is not None:

        d_columns_eval =  [i for i in df_eval.columns if i.startswith('d_')]

    

    calendar = calendar[['d', 'date']]



    # Train

    train_serie = df_train.iloc[[index],:]

    train_serie = pd.melt(train_serie, id_vars=id_columns,

                          value_vars=d_columns_train)

    train_serie.columns = id_columns + ['d', 'sales']

    train_serie = train_serie.merge(calendar, on='d', how='left')

    # Eval

    if df_eval is not None:

        eval_serie = df_eval.iloc[[index],:]

        eval_serie = pd.melt(eval_serie, id_vars=id_columns,

                             value_vars=d_columns_eval)

        eval_serie.columns = id_columns + ['d', 'sales']

        eval_serie = eval_serie.merge(calendar, on='d', how='left')

    # Pred

    if preds is not None:

        pred_serie = pd.concat([eval_serie[['date']],

                                pd.DataFrame(preds[index, :].ravel(),

                                             columns=['sales'])],

                               axis=1)



    fig = go.Figure()

    fig.add_trace(go.Scatter(x=train_serie.date,

                             y=train_serie['sales'],

                             name="train",

                             line_color='deepskyblue'))

    if df_eval is not None:

        fig.add_trace(go.Scatter(x=eval_serie.date,

                                 y=eval_serie['sales'],

                                 name="eval",

                                 line_color='dimgray'))

    if preds is not None:

        fig.add_trace(go.Scatter(x=pred_serie.date,

                                 y=pred_serie['sales'],

                                 name="pred",

                                 line_color='darkmagenta'))



    fig.update_layout(title_text='Time Series: '+ df_train.iloc[index,0],

                      xaxis_rangeslider_visible=True)

    fig.show()
class WRMSSEEvaluator(object):

    

    """

    From WRMSSE Evaluator with extra feature

    https://www.kaggle.com/dhananjay3/wrmsse-evaluator-with-extra-features

    

    """

    

    group_ids = ( 'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',

        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],

        ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])



    def __init__(self, 

                 train_df: pd.DataFrame, 

                 valid_df: pd.DataFrame, 

                 calendar: pd.DataFrame, 

                 prices: pd.DataFrame):

        '''

        intialize and calculate weights

        '''

        self.calendar = calendar

        self.prices = prices

        self.train_df = train_df

        self.valid_df = valid_df

        self.train_target_columns = [i for i in self.train_df.columns if i.startswith('d_')]

        self.weight_columns = self.train_df.iloc[:, -28:].columns.tolist()



        self.train_df['all_id'] = "all"



        self.id_columns = [i for i in self.train_df.columns if not i.startswith('d_')]

        self.valid_target_columns = [i for i in self.valid_df.columns if i.startswith('d_')]



        if not all([c in self.valid_df.columns for c in self.id_columns]):

            self.valid_df = pd.concat([self.train_df[self.id_columns], self.valid_df],

                                      axis=1, 

                                      sort=False)

        self.train_series = self.trans_30490_to_42840(self.train_df, 

                                                      self.train_target_columns, 

                                                      self.group_ids)

        self.valid_series = self.trans_30490_to_42840(self.valid_df, 

                                                      self.valid_target_columns, 

                                                      self.group_ids)

        self.weights = self.get_weight_df()

        self.scale = self.get_scale()

        self.train_series = None

        self.train_df = None

        self.prices = None

        self.calendar = None



    def get_scale(self):

        '''

        scaling factor for each series ignoring starting zeros

        '''

        scales = []

        for i in tqdm(range(len(self.train_series))):

            series = self.train_series.iloc[i].values

            series = series[np.argmax(series!=0):]

            scale = ((series[1:] - series[:-1]) ** 2).mean()

            scales.append(scale)

        return np.array(scales)

    

    def get_name(self, i):

        '''

        convert a str or list of strings to unique string 

        used for naming each of 42840 series

        '''

        if type(i) == str or type(i) == int:

            return str(i)

        else:

            return "--".join(i)

    

    def get_weight_df(self) -> pd.DataFrame:

        """

        returns weights for each of 42840 series in a dataFrame

        """

        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()

        weight_df = self.train_df[["item_id", "store_id"] + self.weight_columns].set_index(

            ["item_id", "store_id"]

        )

        weight_df = (

            weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"})

        )

        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)

        weight_df = weight_df.merge(

            self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"]

        )

        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]

        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)[

            "value"

        ]

        weight_df = weight_df.loc[

            zip(self.train_df.item_id, self.train_df.store_id), :

        ].reset_index(drop=True)

        weight_df = pd.concat(

            [self.train_df[self.id_columns], weight_df], axis=1, sort=False

        )

        weights_map = {}

        for i, group_id in enumerate(tqdm(self.group_ids, leave=False)):

            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)

            lv_weight = lv_weight / lv_weight.sum()

            for i in range(len(lv_weight)):

                weights_map[self.get_name(lv_weight.index[i])] = np.array(

                    [lv_weight.iloc[i]]

                )

        weights = pd.DataFrame(weights_map).T / len(self.group_ids)



        return weights



    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):

        '''

        transform 30490 sries to all 42840 series

        '''

        series_map = {}

        for i, group_id in enumerate(tqdm(self.group_ids, leave=False, disable=dis)):

            tr = df.groupby(group_id)[cols].sum()

            for i in range(len(tr)):

                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values

        return pd.DataFrame(series_map).T

    

    def get_rmsse(self, valid_preds) -> pd.Series:

        '''

        returns rmsse scores for all 42840 series

        '''

        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)

        rmsse = (score / self.scale).map(np.sqrt)

        return rmsse



    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:

        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape



        if isinstance(valid_preds, np.ndarray):

            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)



        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds],

                                axis=1, 

                                sort=False)

        valid_preds = self.trans_30490_to_42840(valid_preds, 

                                                self.valid_target_columns, 

                                                self.group_ids, 

                                                True)

        self.rmsse = self.get_rmsse(valid_preds)

        self.contributors = pd.concat([self.weights, self.rmsse], 

                                      axis=1, 

                                      sort=False).prod(axis=1)

        return np.sum(self.contributors)
id_columns = [i for i in df_train_val.columns if not i.startswith('d_')]

d_columns_train =  [i for i in df_train.columns if i.startswith('d_')]

d_columns_val =  [i for i in df_val.columns if i.startswith('d_')]
train_fold_df = df_train.copy()

valid_fold_df = df_val.copy()

error_eval = WRMSSEEvaluator(train_fold_df, valid_fold_df[d_columns_val], df_calendar, df_price)



l = list([train_fold_df, valid_fold_df])

del l
class M5model(object):

    """

    Generic class for representing M5 Benchmark statistical models 

    """

    

    def __init__(self, horizon):

        """

        horizon : integer, horizon of prediction.

        """

        self.horizon = horizon

        

    def _remove_starting_zeros(self, serie):

        """

        Remove starting zeros from serie

        

        """

        start_index = np.argmax(serie!=0)

        return serie[start_index:]

        

    def predict(self, serie):

        pass

    

        

    def predict_all(self, df_train):

        """

        Predict using the Naive (persistence) method on a DataFrame of time series.

        

        Parameters

        ----------

        df_train : pd.DataFrame, shape (nb_series, ids+d_) d_{i} columns contains sales

        

        Returns

        -------

        preds : array, shape (nb_series, horizon)

            Returns predicted values.

        """

        nb_series = df_train.shape[0]

        preds = np.zeros((nb_series,self.horizon))

        d_columns =  [i for i in df_train.columns if i.startswith('d_')]



        for index, row in enumerate(tqdm(df_train[d_columns].itertuples(index=False), total=len(df_train))):

            row = np.array(row) 

            series = self._remove_starting_zeros(row)

            preds[index,:] = self.predict(series)

        return preds

    

    def create_submission_file(self, df, file_name):

        """

        Create submission file with the predictions

        NB: We double the horizon to take into accoint validation & evaluation forcasts as requested in the submission file

        """

        

        single_horizon = self.horizon

        # double horizon to take into accoint validation & evaluation forcast in the submission file

        self.horizon = 2 * single_horizon

        

        preds = self.predict_all(df)

        

        sample_submission = pd.read_csv(cnt.SAMPLE_SUBMISSION)

        sample_submission.iloc[0:preds.shape[0],1:] = preds[:,0:single_horizon]

        sample_submission.iloc[-preds.shape[0]:,1:] = preds[:,-single_horizon:]



        sample_submission.to_csv(file_name, index=False, compression='gzip')
class M5Naive(M5model):

    """

    Naive (persistence) method for time series forecasting.

    Last known value will be persisted.

    

    """



    def predict(self, serie):

        """

        Predict using the Naive (persistence) method.

        

        Returns

        -------

        predictions : array, shape (horizon,)

            Returns predicted values.

        """

        last_value = serie[-1]

        predictions = np.ones(self.horizon) * last_value

        return predictions
naive_preds_val = M5Naive(horizon=cnt.HORIZON).predict_all(df_train)

naive_error = error_eval.score(naive_preds_val)

naive_error
naive_preds_val.shape
naive_preds_val[:,0:cnt.HORIZON].shape, naive_preds_val[:,-cnt.HORIZON:].shape
plot_time_series(index=24, df_train=df_train, df_eval=df_val,

                 preds=naive_preds_val, calendar=df_calendar)
class M5SeasonalNaive(M5model):

    """

    Seasinal Naive (persistence) method for time series forecasting.

    Last known values in the given seasonal perdiod (expressed in number of days) will be persisted.

    

    """    

    

    def __init__(self, horizon, seasonal_days):

        """

        Initialization

        

        Parameters

        ----------

        horizon : integer, horizon of prediction.

        seasonal_days: int, number of day determining the series seasonality (ex: 7 for weekly)

        """

        self.horizon = horizon

        self.seasonal_days = seasonal_days

        



    def predict(self, sequence):

        

        """

        Predict using the Seasonal Naive method.

        

        Returns

        -------

        predictions : array, shape (horizon,)

            Returns predicted values.

        """

        last_seasonal_values = sequence[-self.seasonal_days:]

        predictions = np.tile(last_seasonal_values,

                              math.ceil(self.horizon/self.seasonal_days))[:self.horizon]

        

        return predictions
snaive_preds_val = M5SeasonalNaive(horizon=cnt.HORIZON,

                                   seasonal_days=7).predict_all(df_train)

snaive_error = error_eval.score(snaive_preds_val)

snaive_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val,

                 preds=snaive_preds_val, calendar=df_calendar)
class M5SimpleExponentialSmoothing(M5model):

    """

    Simple Exponential Smooting method for time series forecasting.

    

    """

    

    def __init__(self, horizon=1, alpha=.1,

                 optimized=False, bounds=(0,1),

                 maxiter=3):

        """

        Params:

        ----------

        

        horizon : integer, horizon of prediction.

        alpha : float,Exponential smoothing parameter, range(0,1)

        optimized: boolean, if True alpha is calculated and optimized automatically

        bounds: 2D-tuple, (lower_bound, upper_bound) for alpha param

        maxiter: int, max number of iteration for finding the optimal alpha (the higher the more accurate, but also the slower)

        """

        self.horizon = horizon

        self.alpha = alpha

        self.optimized = optimized

        self.bounds = bounds

        self.maxiter = maxiter

        

    def _fit(self, ts, alpha):

        """

        Fit Simple Exponential Smoothing 

        """

        len_ts = len(ts)

        es = np.zeros(len_ts) # exponential-smoothing array

        

        # init

        es[0] = ts[0]

        

        for i in range(1, len_ts):

            es[i] = alpha * ts[i-1] + (1-alpha)*es[i-1]

            

        return es

    

    def _mse(self, ts, alpha):

        

        es = self._fit(ts, alpha)

        

        mse = np.mean(np.square(ts - es))

        

        return mse

    

    def _best_alpha(self, ts):

        """

        Calculate best alpha parameter based on MSE

        """

        

        res = minimize_scalar(lambda alpha: self._mse(ts, alpha),

                              bounds=self.bounds,

                              method='bounded',

                              options={'xatol': 1e-05,

                                       'maxiter': self.maxiter}

                             )

        

        return res.x

    

    def predict(self, ts):

        """

        Predict with Simple Exponential Smoothing method

        

        Parameters

        ----------

        ts : array, time series array

        

        Returns

        -------

        preds : array, shape (horizon,)

            Returns predicted values.

        """

        if self.optimized:

            alpha = self._best_alpha(ts)

            self.alpha = alpha

        

        len_ts = len(ts)

        es = np.zeros(len_ts) # exponential-smoothing array

        

        # init

        es[0] = ts[0]

        

        for i in range(1, len_ts):

            es[i] = self.alpha * ts[i] + (1-self.alpha)*es[i-1]

        

        preds = np.repeat(es[-1], self.horizon)

        

        return preds
ses_preds_val = M5SimpleExponentialSmoothing(horizon=cnt.HORIZON, alpha=.1,

                                             optimized=False, bounds=(0.1,.3),

                                             maxiter = 1).predict_all(df_train)

ses_error = error_eval.score(ses_preds_val)

ses_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val,

                 preds=ses_preds_val, calendar=df_calendar)
class M5MovingAverage(M5model):

    

    """

    Moving Average method for time series forecasting.

    

    """

    

    def __init__(self, horizon, k, optimized=False,

                 k_lb=2, k_ub=5, last_n_values=None):

        """

        horizon : integer, horizon of prediction.

        k : integer, moving average is calculated from the moving last k elements 

        optimized boolean, if True parameter k is calculated and optimized automatically

        k_lb :  integer, lower bound of k paramter

        k_ub : integer, upper bound of k paramter

        last_n_values : int, default None, take last n values of the serie to calculate best k param (to speed up)



        """

        self.horizon = horizon

        self.k = k

        self.optimized = optimized

        self.k_lb = k_lb

        self.k_ub = k_ub

        self.last_n_values = last_n_values

        

    

    

    def calculate_best_k_parameter(self, serie):

        """

        Calulate the optimal (in terms of mse) paramter k for moving average.

        Paramter k determines the last k elements of the serie that have to be taken to calculate the (moving) average



        Parameters

        ----------

        serie : array, vector containing the serie's values

        

        Returns

        -------

        best_k : int

               Returns the best k value selected from the range [k_lb, k_ub] by minimizing the insample MSE.

        """

        serie = self._remove_starting_zeros(serie)

        

        if self.last_n_values is not None:

            serie = serie[-self.last_n_values:] #reduce serie to its last_n_values

            

        serie_len = len(serie)

        mse = np.zeros(self.k_ub - self.k_lb +1)

        all_k = list(range(self.k_lb, self.k_ub + 1))

        



        for ind, k in enumerate(all_k):

            moving_average_values = np.zeros((serie_len - k))

            for i in range(k, serie_len):

                sliding_window = serie[i-k:i]

                moving_average_values[i-k] = np.average(sliding_window)



            mse[ind] = np.average(np.square(serie[k:] - moving_average_values))



        best_k = all_k[np.argmin(mse)]

    

        return best_k

    

    def predict(self, serie):

        """

        Predict using the Moving Average method.

        

        Parameters

        ----------

        serie : array, vector of serie values

       

        Returns

        -------

        predictions : array, shape (horizon,)

            Returns predicted values.

        """

        

        if self.optimized:

            

            serie_optimized = serie

            

            if self.last_n_values is not None:

                # To speed up calcuation, best k param is calculated from last_n_values

                serie_optimized = serie_optimized[-self.last_n_values:] 

            best_k = self.calculate_best_k_parameter(serie_optimized)

            self.k = best_k

    

        working_serie = np.concatenate((serie[-self.k:], np.zeros(self.horizon)))

        for i in range(self.horizon):

            working_serie[self.k+i] = np.average(working_serie[i:self.k+i])

        return working_serie[self.k:]

        
%%time

ma_preds_val = M5MovingAverage(k=3, horizon=cnt.HORIZON, optimized=True,

                               k_lb=3, k_ub=5, last_n_values=28).predict_all(df_train)

ma_error = error_eval.score(ma_preds_val)

ma_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val,

                 preds=ma_preds_val, calendar=df_calendar)
class M5Croston(M5model):

    

    def __init__(self, horizon):

        self.horizon = horizon

        self.smoothing_level = .1

        self.optimized = False

        self.maxiter = 3

        self.debiasing = 1

    

    

    def _inter_demand_intervals(self, ts):

        """

        Calculate inter-demand intervals of serie

        """

        

        demand_times = np.argwhere(ts>0).ravel() +1

        a = demand_times - shift(demand_times, 1, cval=0)

        return a

    

    def _positive_demand(self, ts):

        

        """

        Calculates non-zero demand (values) of a serie

        """

        return ts[ts>0]

    

    def predict(self, ts):

        

        """

        Predict using the Croston method.

        

        Parameters

        ----------

        ts : array, vector of time-series values

        horizon : integer, horizon of prediction.

        

        Returns

        -------

        predictions : array, shape (horizon,)

            Returns predicted values.

        """

        

        ts = np.array(ts)

        p = self._inter_demand_intervals(ts)

        a = self._positive_demand(ts)

        

        p_est = M5SimpleExponentialSmoothing(horizon=self.horizon,

                                             alpha=.1,

                                             optimized=self.optimized,

                                             bounds=(0.1,.3)).predict(p)

        a_est = M5SimpleExponentialSmoothing(horizon=self.horizon,

                                             alpha=.1,

                                             optimized=self.optimized,

                                             bounds=(0.1,.3)).predict(a)

      

        # Future Forecast

        future_forecasts = self.debiasing * np.divide(a_est, p_est)

        

        return future_forecasts

    
%%time

cro_preds_val = M5Croston(horizon=cnt.HORIZON).predict_all(df_train)

cro_error = error_eval.score(cro_preds_val)

cro_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val,

                 preds=cro_preds_val, calendar=df_calendar)
class M5OptCroston(M5Croston):

    """

    Optimized Croston model

    """

    

    def __init__(self, horizon, maxiter):

        super().__init__(horizon)

        self.optimized = True

        self.maxiter = maxiter
%%time

optcro_preds_val = M5OptCroston(horizon=cnt.HORIZON,

                                maxiter=3).predict_all(df_train)

optcro_error = error_eval.score(optcro_preds_val)

optcro_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val,

                 preds=optcro_preds_val, calendar=df_calendar)
class M5SBA(M5Croston):

    """

    Syntetos-Boylan Approximation (SBA) model

    """

    

    def __init__(self, horizon):

        super().__init__(horizon)

        self.debiasing = .95
%%time

sba_preds_val = M5SBA(horizon=cnt.HORIZON).predict_all(df_train)

sba_error = error_eval.score(sba_preds_val)

sba_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val,

                 preds=sba_preds_val, calendar=df_calendar)
class M5TSB(M5model):

    

    """

    Teunter-Syntetos-Babai method (TSB)

    

    Inspired by https://medium.com/analytics-vidhya/croston-forecast-model-for-intermittent-demand-360287a17f5f

    """

    

    def __init__(self, horizon, alpha, beta):

        self.horizon = horizon

        self.alpha = alpha

        self.beta = beta

        

    def predict(self, ts):

        

        ts = np.array(ts) # Transform the input into a numpy array

        len_ts = len(ts) # Historical period length

    

        #level (a), probability(p) and forecast (f)

        a = np.zeros(len_ts+1)

        p = np.zeros(len_ts+1)

        f = np.zeros(len_ts+1)

        

        first_occurence = np.argmax(ts>0)

        a[0] = ts[first_occurence]

        p[0] = 1/(1 + first_occurence)

        f[0] = p[0]*a[0]



        # Create all the t+1 forecasts

        for t in range(0,len_ts): 

            if ts[t] > 0:

                a[t+1] = self.alpha*ts[t] + (1-self.alpha)*a[t] 

                p[t+1] = self.beta*(1) + (1-self.beta)*p[t]  

            else:

                a[t+1] = a[t]

                p[t+1] = (1-self.beta)*p[t]       

            f[t+1] = p[t+1]*a[t+1]



        # Future Forecast

        

        preds = np.repeat(f[-1], self.horizon)

        

        return preds

                      

    
%%time

tsb_preds_val = M5TSB(horizon=cnt.HORIZON,

                      alpha=.1, beta=.1).predict_all(df_train)

tsb_error = error_eval.score(tsb_preds_val)

tsb_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val,

                 preds=tsb_preds_val, calendar=df_calendar)
class M5TopDown(object):

    """

    Base class to model Top-DOwn approach: 

        - predicting top-level time-series and 

        - disaggregating predictions proportional to bottom time-series values

    """

    

    def __init__(self, horizon, df_train, df_calendar):

        self.horizon = horizon

        self.df_train = df_train

        self.df_calendar = df_calendar

        self.top_level = self._get_top_level_timeserie()

    

    def _get_top_level_timeserie(self):

        """

        Calculate top level time series by aggregating low level time-series

        """

        

        data = np.sum(self.df_train.iloc[:,6:].values, axis=0)

        index= pd.date_range(start=self.df_calendar['date'][0],

                             end=self.df_calendar['date'][len(data)-1],

                             freq='D')

        top_level = pd.Series(data, index)

        return top_level

    

    def _get_weights(self):

        """

        Claculate weights based on the last 28 days for each time series;

        These weights will be used to disaggregate the top level time-series

        """

        w = np.sum(self.df_train.iloc[:,-28:].values, axis=1) / sum(self.top_level[-28:])

        w = w.reshape(len(w),1)

        return w

    

    def _parameters_tuning(self):

        """

        Tuning hyper-parameters of the model used for predicting the top-level time-series future horizon.

        The implenetation will depend on the method used (ex: Expontial Smoothign, ARIMA, etc)

        """

        pass

    

    def predict_top_level(self):

        """

        Predict the future horizon of top level time-series

        """

        pass

    

    def predict_bottom_levels(self):

        """

        Predict the future horizon of the bottom level time-series by disaggregating the the top level predictions

        """

        

        w = self._get_weights()

        

        top_level_preds = self.predict_top_level().values

        top_level_preds = top_level_preds.reshape(1,len(top_level_preds))

        

        preds = np.multiply(top_level_preds, w)

        

        return preds

    

    

    def create_submission_file(self, file_name):

        """

        Create submission file with the predictions

        NB: We double the horizon to take into account validation & evaluation forcasts as requested in the submission file

        """

        

        single_horizon = self.horizon

        # double horizon to take into accoint validation & evaluation forcast in the submission file

        self.horizon = 2 * single_horizon

        

        preds = self.predict_bottom_levels()

        

        sample_submission = pd.read_csv(cnt.SAMPLE_SUBMISSION)

        sample_submission.iloc[0:preds.shape[0],1:] = preds[:,0:single_horizon]

        sample_submission.iloc[-preds.shape[0]:,1:] = preds[:,-single_horizon:]



        sample_submission.to_csv(file_name, index=False, compression='gzip')

        

    
class M5ExponentialSmoothing(M5TopDown):

    """

    An algorithm is used to select the most appropriate exponential smoothing model

    for predicting the top level of the hierarchy (level 1 of Table 1), indicated through information criteria (AIC).

    

    The top-down method will be used for obtaining reconciled forecasts

    at the rest of the hierarchical levels (based on historical proportions, estimated for the last 28 days).

    """

    

    def __init__(self, horizon, df_train, df_calendar):

        self.horizon = horizon

        self.df_train = df_train

        self.df_calendar = df_calendar

        self.top_level = self._get_top_level_timeserie()

    

    def _parameters_tuning(self):

        

        # prepare param grid

        trend_param = ['add', 'mul', None]

        seasonal_param = ['add', 'mul', None]

        damped_param=[True, False]

        

        params = [trend_param, seasonal_param, damped_param]

        grid_param = list(itertools.product(*params))

        grid_param = [(trend, seasonal, damped) for trend, seasonal, damped in grid_param if not(trend==None and damped==True)]

        

        aic = np.ones(len(grid_param))*np.nan

        

        # grid-search

        for i, (trend, seasonal, damped) in enumerate(grid_param):

            

            ES = ExponentialSmoothing(self.top_level, trend=trend,

                                      seasonal=seasonal, damped=damped, 

                                      seasonal_periods=7,freq='D').fit(optimized=True, use_brute=True)

            aic[i] = ES.aic

        

        # best parameters & AIC

        best_index = np.nanargmin(aic)

        best_params = grid_param[best_index]

        best_aic = aic[best_index]

        

        return best_params, best_aic

    

    def predict_top_level(self):

        

        

        (best_trend, best_seas, best_dumped), best_aic = self._parameters_tuning()

        

        ES = ExponentialSmoothing(self.top_level, trend=best_trend,

                                  damped=best_dumped, seasonal=best_seas,

                                  seasonal_periods=7, freq='D').fit(optimized=True, use_brute=True)

        top_level_preds = ES.forecast(self.horizon)

        

        return top_level_preds

        
%%time

es_preds_val = M5ExponentialSmoothing(horizon=cnt.HORIZON, df_train=df_train, df_calendar=df_calendar).predict_bottom_levels()

es_error = error_eval.score(es_preds_val)

es_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val, preds=es_preds_val, calendar=df_calendar)
class M5ARIMA(M5TopDown):

    

    def __init__(self, horizon, df_train, df_calendar):

        self.horizon = horizon

        self.df_train = df_train

        self.df_calendar = df_calendar

        self.top_level = self._get_top_level_timeserie()

        self.exog_fit = None

        self.exog_pred = None

    

    def _get_aic(self, order):

        """

        Because some parameter combinations may lead to numerical misspecifications,

        we explicitly disabled warning messages in order to avoid an overload of warning messages.

        These misspecifications can also lead to errors and throw an exception,

        so we make sure to catch these exceptions and ignore the parameter combinations that cause these issues.

        """

        aic =np.nan

        try:

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore",

                                        message="Maximum Likelihood optimization failed to converge. Check mle_retvals")



                arima_mod=sm.tsa.statespace.SARIMAX(endog=self.top_level, exog=self.exog_fit,

                                                    order=order,

                                                    enforce_stationarity=False, 

                                                    enforce_invertibility=False).fit()

                aic=arima_mod.aic

        except:

            pass

        return aic



    def _parameters_tuning(self,n_jobs=7):

        d = range(3)

        p = range(12)

        q = range(12)



        params = [p, d, q]

        pdq_params = list(itertools.product(*params))



        get_aic_partial=partial(self._get_aic)

        p = Pool(n_jobs)

        res_aic = p.map(get_aic_partial, pdq_params)  

        p.close()

        

        best_aic_index = np.nanargmin(res_aic)

        best_aic = res_aic[best_aic_index]

        best_pdq = pdq_params[best_aic_index]

        

        return best_pdq, best_aic

    

    def predict_top_level(self):

        

        

        best_pdq, best_aic = self._parameters_tuning()

        

        arima_mod = sm.tsa.statespace.SARIMAX(endog=self.top_level, exog=self.exog_fit,

                                              order=best_pdq,

                                              enforce_stationarity=False, 

                                              enforce_invertibility=False).fit()

        top_level_preds = arima_mod.forecast(self.horizon, exog=self.exog_pred)

        

        return top_level_preds

        
%%time

arima_preds_val = M5ARIMA(horizon=cnt.HORIZON, df_train=df_train, df_calendar=df_calendar).predict_bottom_levels()

arima_error = error_eval.score(arima_preds_val)

arima_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val, preds=arima_preds_val, calendar=df_calendar)
class M5ARIMAX(M5ARIMA):

    

    def __init__(self, horizon, df_train, df_calendar):

        self.horizon = horizon

        self.df_train = df_train

        self.df_calendar = df_calendar

        self.train_len = len([x for x in self.df_train.columns if x.startswith('d_')])

        self.top_level = self._get_top_level_timeserie()

        self.exog_fit, self.exog_pred = self._get_exploratory_variables()

        

        

        

    def _get_exploratory_variables(self):

        calendar = self.df_calendar[['date', 'snap_CA','snap_TX','snap_WI', 'event_name_1', 'event_name_2']].copy()

        calendar['snap_count'] = calendar[['snap_CA','snap_TX','snap_WI']].apply(np.sum,axis=1)



        calendar['is_event_1'] = [isinstance(x , str)*1 for x in calendar['event_name_1']]

        calendar['is_event_2'] = [isinstance(x , str)*1 for x in calendar['event_name_2']]

        calendar['is_event'] = calendar[['is_event_1', 'is_event_2']].apply(np.sum, axis=1)

        calendar['is_event'] = np.where(calendar['is_event']>0,1,0)



        exog_fit = calendar[['snap_count', 'is_event']].iloc[:self.train_len,:].values

        exog_pred = calendar[['snap_count', 'is_event']].iloc[self.train_len:self.train_len+self.horizon,:].values



        return exog_fit, exog_pred

    

    def create_submission_file(self, file_name):

        """

        Create submission file with the predictions

        NB: We double the horizon to take into account validation & evaluation forcasts as requested in the submission file

        """

        

        single_horizon = self.horizon

        # double horizon to take into accoint validation & evaluation forcast in the submission file

        #self.horizon = 2 * single_horizon

        

        # Ri-Calculate Ex Variables to take into account 2*horizon

        self.exog_fit, self.exog_pred = self._get_exploratory_variables()

        

        preds = self.predict_bottom_levels()

        

        sample_submission = pd.read_csv(cnt.SAMPLE_SUBMISSION)

        sample_submission.iloc[0:preds.shape[0],1:] = preds[:,0:single_horizon]

        sample_submission.iloc[-preds.shape[0]:,1:] = preds[:,-single_horizon:]



        sample_submission.to_csv(file_name, index=False, compression='gzip')

        
%%time

arimax_preds_val = M5ARIMAX(horizon=cnt.HORIZON, df_train=df_train, df_calendar=df_calendar).predict_bottom_levels()

arimax_error = error_eval.score(arimax_preds_val)

arimax_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val, preds=arimax_preds_val, calendar=df_calendar)
from fbprophet import Prophet



class M5Prophet(M5ARIMAX):

    

    def __init__(self, horizon, df_train, df_calendar):

        self.horizon = horizon

        self.df_train = df_train

        self.df_calendar = df_calendar

        self.train_len = len([x for x in self.df_train.columns if x.startswith('d_')])

        self.top_level = self._get_top_level_timeserie()

        self.exog_fit, self.exog_pred = self._get_exploratory_variables()

        

    

    def predict_top_level(self):

        

        df = pd.DataFrame({'ds':self.top_level.index, 'y':self.top_level.values})

        df['snap_count'] = self.exog_fit[:,0]

        df['is_event'] = self.exog_fit[:,1]

        

        m = Prophet()

        m.add_regressor('snap_count')

        m.add_regressor('is_event')

        

        m.add_country_holidays(country_name='US')

        m.fit(df)

        

        future = m.make_future_dataframe(periods=self.horizon)

        future['snap_count'] = np.concatenate((self.exog_fit[:,0],self.exog_pred[:,0]))

        future['is_event'] = np.concatenate((self.exog_fit[:,1], self.exog_pred[:,1]))

        

        preds = m.predict(future)

        

        top_level_preds = pd.Series(preds['yhat'].values[-self.horizon:],

                                    index = preds['ds'].values[-self.horizon:])

        

        return top_level_preds
%%time

fp_preds_val = M5Prophet(horizon=cnt.HORIZON, df_train=df_train, df_calendar=df_calendar).predict_bottom_levels()

fp_error = error_eval.score(fp_preds_val)

fp_error
plot_time_series(index=24, df_train=df_train, df_eval=df_val, preds=fp_preds_val, calendar=df_calendar)
method = ['Naive', 'sNaive', 'SES', 'MA', 'CRO', 'optCRO', 'SBA','TSB', 'ES', 'ARIMA', 'ARIMAX', 'prophet']

error = [naive_error, snaive_error, ses_error, ma_error, cro_error, optcro_error,

         sba_error, tsb_error, es_error, arima_error, arimax_error, fp_error]

validation_errors = pd.DataFrame({'method':method, 'WRMSSE':error}).sort_values('WRMSSE').reset_index(drop=True)



validation_errors
%%time

M5Naive(horizon=cnt.HORIZON).create_submission_file(df_train_val,

                                                    file_name='submission_naive.csv.gz')
%%time

M5SeasonalNaive(horizon=cnt.HORIZON, seasonal_days=7).create_submission_file(df_train_val,

                                                                             file_name='submission_snaive.csv.gz')
%%time

M5SimpleExponentialSmoothing(horizon=cnt.HORIZON, alpha=.1,

                             optimized=True, bounds=(0.1,.3),

                             maxiter = 10).create_submission_file(df_train_val,

                                                                  file_name='submission_ses.csv.gz')
%%time

M5MovingAverage(k=3, horizon=cnt.HORIZON, optimized=True,

                k_lb=3, k_ub=5, last_n_values=100).create_submission_file(df_train_val,

                                                                          file_name='submission_ma.csv.gz')
%%time

M5Croston(horizon=cnt.HORIZON).create_submission_file(df_train_val,

                                                      file_name='submission_cro.csv.gz')
%%time

M5OptCroston(horizon=cnt.HORIZON, maxiter=10).create_submission_file(df_train_val,

                                                                     file_name='submission_optcro.csv.gz')
%%time

M5SBA(horizon=cnt.HORIZON).create_submission_file(df_train_val,

                                                  file_name='submission_sba.csv.gz')
%%time

M5TSB(horizon=cnt.HORIZON, alpha=.1, beta=.1).create_submission_file(df_train_val,

                                                                     file_name='submission_tsb.csv.gz')
%%time

M5ExponentialSmoothing(horizon=cnt.HORIZON,

                       df_train=df_train_val,

                       df_calendar=df_calendar).create_submission_file(file_name='submission_es.csv.gz')
%%time

M5ARIMA(horizon=cnt.HORIZON,

        df_train=df_train_val,

        df_calendar=df_calendar).create_submission_file("submission_arima.csv.gz")
%%time

M5ARIMAX(horizon=cnt.HORIZON,

        df_train=df_train_val,

        df_calendar=df_calendar).create_submission_file("submission_arimax.csv.gz")
M5Prophet(horizon=cnt.HORIZON,

          df_train=df_train,

          df_calendar=df_calendar).create_submission_file("submission_prophet.csv.gz")
top_methods = validation_errors['method'].head(3).values

submission_files = ['submission_'+method.lower()+'.csv.gz' for method in top_methods]
all_preds = np.zeros((60980, cnt.HORIZON, len(submission_files)))



for i, file in tqdm(enumerate(submission_files)):

    sub_df = pd.read_csv(file)

    all_preds[:,:,i] = sub_df.iloc[:,1:].values
final_pred = np.mean(all_preds, axis=2)
final_pred.shape
final_submission = pd.read_csv(cnt.SAMPLE_SUBMISSION)

final_submission.iloc[0:final_pred.shape[0],1:] = final_pred



final_submission.to_csv("submission.csv.gz", index=False, compression='gzip')
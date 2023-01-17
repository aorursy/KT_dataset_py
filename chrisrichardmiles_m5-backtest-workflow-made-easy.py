#### CODE IN THE THIS CELL WAS CREATED BY SAKAMI
from typing import Union

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm_notebook as tqdm 


class WRMSSEEvaluator(object):
    """
    Example usage:
    train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
train_fold_df = train_df.iloc[:, :-28]
valid_fold_df = train_df.iloc[:, -28:]
valid_preds = valid_fold_df.copy() + np.random.randint(100, size=valid_fold_df.shape)

evaluator = WRMSSEEvaluator(train_fold_df, valid_fold_df, calendar, prices)
evaluator.score(valid_preds)
"""

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        
        # I am adding just this one line to this implementation to avoid changing the train_df globally, which happened to be in my backtest object
        train_df = train_df.copy()
        # And this to keep track of scores by level
        self.all_scores = []
        
        
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices
        

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            setattr(self, f'lv{i + 1}_train_df', train_df.groupby(group_id)[train_target_columns].sum())
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        train_y = getattr(self, f'lv{lv}_train_df')
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = ((train_y.iloc[:, 1:].values - train_y.iloc[:, :-1].values) ** 2).mean(axis=1)
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            self.all_scores.append(lv_scores.sum())

        return np.mean(self.all_scores)

import pandas as pd
import numpy as np 

class Backtest:
    """
    Validates your prediction process by running it on different time periods, 
    with the last 28 days of the time period used as the validation set.
    This object will store the final scores in the .scores attribute
    """
    def __init__(self, cal, stv, s_p, process: list, process_names=['process'], verbose=False): 
        self.cal = cal
        self.stv = stv
        self.s_p = s_p
        self.process = process
        self.process_names = process_names
        self.verbose = verbose
        self.valid_end_days = []
        self.scores = {name: [] for name in process_names} 
        self.preds = [[] for name in process_names]
        self.evaluator = None
        

    def score(self, days_back=0): 
        """
        gives the score for your predictions if the data ended days_back[n] days ago
        """
        if days_back != 0: 
            stv = self.stv.iloc[:, :-days_back]
        else:   
            stv = self.stv
        train_df = stv.iloc[:, :-28]
        valid_df = stv.iloc[:, -28:]
        self.valid_end_days.append(1913-days_back)
        
        self.evaluator = WRMSSEEvaluator(train_df, valid_df, self.cal, self.s_p)
        
        for i in range(len(self.process)):
            valid_preds = self.process[i](self.cal, train_df, self.s_p)
            score = self.evaluator.score(valid_preds)
            self.scores[self.process_names[i]].append(score)
            self.preds[i].append(valid_preds)
            
            
            if self.verbose == True: 
                print(f'{self.process_names[i]} had a score of {score} on validation period {1913-days_back-28} to {1913-days_back}')
        
    def score_all(self, days_back=[0,308]):
        for i in range(len(days_back)): 
            self.score(days_back[i])
            
    
            
    def scores_df(self):
        return pd.DataFrame(self.scores, index=self.valid_end_days)
    
    
            
    
# for plotting later
import matplotlib.pyplot as plt
PATH = '/kaggle/input/m5-forecasting-accuracy/'
cal = pd.read_csv(f'{PATH}calendar.csv')
sell_prices = pd.read_csv(f'{PATH}sell_prices.csv')
ss = pd.read_csv(f'{PATH}sample_submission.csv')
stv = pd.read_csv(f'{PATH}sales_train_validation.csv')
def add_snap_col(df_in):
    """adds a 'snap_day' column to a dataframe that contains a state_id column and the columns 'snap_CA', 'snap_TX', 
    and snap_WI"""
    df = pd.get_dummies(df_in, columns=['state_id'])
    df['snap_day'] = (df.snap_CA * df.state_id_CA) + (df.snap_WI * df.state_id_WI) + (df.snap_TX * df.state_id_TX)
    del df['state_id_WI'], df['state_id_CA'], df['state_id_TX']
    return df



def melt_merge_snap(df):
    df = df.melt(['id', 'state_id'], var_name='d', value_name='demand')
    df = df.merge(cal)
    df = add_snap_col(df)
    return df

def get_valid(stv, n):
    """gets a df for the next 28 days to give predictions"""
    valid = stv.iloc[:, pd.np.r_[0,5,-28:0]]
    valid.columns = ['id', 'state_id'] + ['d_' + str(n+x) for x in range(1,29)]
    return valid
    
def join_valid_groupby(valid, group):
    """ 
    Joins the sub dataframe created by get_valid to a groupby series
    """
    return valid.join(group, on=group.index.names)

def reshape_valid(df): 
    """Takes a dataframe in the form given by join_valid_groupby, or any dataframe with the proper index and and 'd' colums.
    returns a prediction dataframe that can be input into an evaluator
    """
    # pivot df to get it into the proper format for submission
    df = df.pivot(index='id', columns='d', values='demand')
    # need to reset index to take care of columns. comment next line out to see what i mean 
    #df.set_index('id',inplace=True)
    return df#.iloc[:, -28:]
def process_00(cal, stv, sell_prices, days_to_agg=28):
    """a process that produces predictions for the next 28 days following the end of stv"""
    
    d = 1913 - (1919 - stv.shape[1])
    last_90 = stv.iloc[:, pd.np.r_[0,5,-days_to_agg:0]].copy() # we include 0, and 5 to get the id and state id columns
    last_90 = melt_merge_snap(last_90)
    by_weekday_snap_90 = last_90.groupby(['id', 'wday', 'snap_day'])['demand'].mean()
    valid = get_valid(stv, d)
    valid_id = valid[['id']]
    valid = melt_merge_snap(valid)
    valid.drop('demand', axis='columns', inplace=True)
    valid = join_valid_groupby(valid, by_weekday_snap_90)
    preds = reshape_valid(valid).reset_index()
    return valid_id.merge(preds).iloc[:, 1:]
def process_01(cal, stv, sell_prices):
    return process_00(cal, stv, sell_prices, days_to_agg=60)

def process_02(cal, stv, sell_prices):
    return process_00(cal, stv, sell_prices, days_to_agg=90)

def process_03(cal, stv, sell_prices):
    return process_00(cal, stv, sell_prices, days_to_agg=120)

def process_04(cal, stv, sell_prices):
    return process_00(cal, stv, sell_prices, days_to_agg=150)
backtest = Backtest(cal, stv, sell_prices, 
                    process=[process_02], 
                    process_names=['agg_90'])
backtest = Backtest(cal, stv, sell_prices, 
                    process=[process_00, process_01, process_02, process_03, process_04], 
                    process_names=['agg_28', 'agg_60', 'agg_90', 'agg_120', 'agg_150'])
%%time
backtest.score_all([0, 28, 56, 84, 112, 140, 168, 196, 224, 252, 280, 308, 336, 364,
                   392, 420, 448, 476, 504, 532, 560, 588, 616, 644, 672, 700, 728])
backtest.score(0)

scores_df = backtest.scores_df()
scores_df.head()
e = backtest.evaluator
preds = backtest.preds
preds[0][0] * 1.04
levels = [
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
]
level_scores = zip(levels, e.all_scores)
list(level_scores)

e1 = backtest.evaluator
e1.all_scores = []
e.score(preds[0][0])

import matplotlib.pyplot as plt
scores_df.plot(figsize=(20,7))
plt.xlabel('Validation end day', fontsize=20)
plt.ylabel('WRMSSEE score', fontsize=20)
plt.title('Process performance over time', fontsize=26)
plt.show()
scores_df.to_csv('backtest_results_' + '_'.join(backtest.process_names) + '.csv')


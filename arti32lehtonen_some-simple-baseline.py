from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm


class WRMSSEEvaluator(object):
    # https://www.kaggle.com/dhananjay3/wrmsse-evaluator-with-extra-features    
    group_ids = (
        'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],
        ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id']
    )

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

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
%matplotlib inline
dates_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
prices_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sales_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
sales_df.head()
sales_array = sales_df.iloc[:, 6:].values
val_days_amount = 28
valid_evaluator = WRMSSEEvaluator(
    sales_df.iloc[:, :-val_days_amount],
    sales_df.iloc[:, -val_days_amount:],
    dates_df,
    prices_df
)
def get_weighted_moving_average_prediction_online(train_sales, beta=0.5):
    result = train_sales[:, 0]
    for n in range(1, train_sales.shape[1]):
        result = beta * train_sales[:, n] + (1 - beta) * result
    return result
X_train = sales_array[:, :-val_days_amount]
X_val = sales_array[:, -val_days_amount:]

first_val_day_index = int(sales_df.columns[-val_days_amount].partition('_')[2])
all_wrmse_scores = []
day_columns = [f'd_{i}' for i in range(first_val_day_index, first_val_day_index + val_days_amount)]

for beta in tqdm(np.linspace(0.1, 0.9, 9), leave=False):
    result = get_weighted_moving_average_prediction_online(X_train, beta)
    result = np.array([result] * val_days_amount).T
    result_df = pd.DataFrame(result, columns=day_columns)
    wrmse_score = valid_evaluator.score(result_df)
    all_wrmse_scores.append(wrmse_score)
    
    del result_df
plt.plot(np.linspace(0.1, 0.9, 9), all_wrmse_scores)
plt.xlabel(r'$\beta$')
plt.ylabel('wrmsse')
all_wrmse_scores = []
day_columns = [f'd_{i}' for i in range(first_val_day_index, first_val_day_index + val_days_amount)]


for beta in np.linspace(0.01, 0.1, 5):
    result = get_weighted_moving_average_prediction_online(X_train, beta)
    result = np.array([result] * val_days_amount).T
    result_df = pd.DataFrame(result, columns=day_columns)
    wrmse_score = valid_evaluator.score(result_df)
    all_wrmse_scores.append(wrmse_score)
    
    del result_df
plt.plot(np.linspace(0.01, 0.1, 5), all_wrmse_scores)
plt.xlabel(r'$\beta$')
plt.ylabel('wrmsse')
chosen_beta = 0.05
def get_weighted_moving_average_prediction_online_with_season(
    train_sales, beta=0.5, season_d=7, gamma=0.1,
):
    result = train_sales[:, 0]
    season_result = [train_sales[:, i] for i in range(season_d)]
    for n in range(1, train_sales.shape[1]):
        result = beta * train_sales[:, n] + (1 - beta) * result
        if n >= season_d:
            season_result[n % season_d] = gamma * train_sales[:, n] + (1 - gamma) * season_result[n % season_d]
    return result, season_result
season_d = 7
result, season_result = get_weighted_moving_average_prediction_online_with_season(
    X_train, beta=0.05, season_d=season_d, gamma=0.25
)
all_wrmse_scores = []
day_columns = [f'd_{i}' for i in range(first_val_day_index, first_val_day_index + val_days_amount)]


for alpha in [0.1, 0.2, 0.3, 0.4]:
    final_result = np.array([result] * val_days_amount).T
    for i in range(final_result.shape[1]):
        season_index = (X_train.shape[1] + i) % season_d
        final_result[:, i] = final_result[:, i] * alpha + (1-alpha) * season_result[season_index]
    result_df = pd.DataFrame(final_result, columns=day_columns)
    wrmse_score = valid_evaluator.score(result_df)
    all_wrmse_scores.append(wrmse_score)
    
    del result_df
plt.plot([0.1, 0.2, 0.3, 0.4], all_wrmse_scores)
plt.xlabel(r'$\alpha$')
plt.ylabel('wrmsse')
alpha = 0.01
beta = 0.05
gamma = 0.25


result, season_result = get_weighted_moving_average_prediction_online_with_season(
    X_train, beta=beta, season_d=7, gamma=gamma
)
val_days_amount = 56


final_result = np.array([result] * val_days_amount).T
for i in range(final_result.shape[1]):
    season_index = (X_train.shape[1] + i) % 7
    final_result[:, i] = final_result[:, i] * alpha + (1-alpha) * season_result[season_index]
n_objects = X_train.shape[0]
day_columns = [f'F{i}' for i in range(1, 29)]
sample_submission.loc[:n_objects - 1, day_columns] = final_result[:, :28]
sample_submission.loc[n_objects:, day_columns] = final_result[:, 28:]
sample_submission.to_csv('/kaggle/working/baseline_ma7days.csv', index=0)
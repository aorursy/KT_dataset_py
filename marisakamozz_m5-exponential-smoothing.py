!pip install statsmodels==0.12.0
from multiprocessing import Pool

import numpy as np

import pandas as pd

from statsmodels.tsa.api import ExponentialSmoothing
# https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists

from itertools import product

def my_product(inp):

    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]



pattern = {

    'trend': [None, 'add'],

    'seasonal': [None, 'add'],

}

params = my_product(pattern)
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'])
def read_sales(filename):

    sales = pd.read_csv(filename)

    sales.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1, inplace=True)

    sales.set_index('id', inplace=True)

    sales.columns = calendar.date[:len(sales.columns)]

    return sales
# metric = 'aic'

metric = 'aicc'

# metric = 'bic'
def fit_es(data):

    data_id, data = data

    first_index = data[data > 0].index[0]

    data = data.loc[first_index:]

    best_score = np.inf

    best_model = None

    for param in params:

        fit = ExponentialSmoothing(data, seasonal_periods=7, initialization_method='estimated', freq='D', **param).fit()

        if metric == 'aic':

            if best_score > fit.aic:

                best_score = fit.aic

                best_model = fit

        elif metric == 'aicc':

            if best_score > fit.aicc:

                best_score = fit.aicc

                best_model = fit

        elif metric == 'bic':

            if best_score > fit.bic:

                best_score = fit.bic

                best_model = fit

        else:

            raise NotImplemntedError()

    f = best_model.forecast(28)

    f = pd.DataFrame([f])

    f.columns = [f'F{i+1}' for i in range(28)]

    f.insert(0, 'id', data_id)

    return f
def forecast(sales):

    # sales_list = list(sales.head(100).iterrows())

    sales_list = list(sales.iterrows())

    pool = Pool(4)

    result = pool.map(fit_es, sales_list)

    return pd.concat(result)
%%time

sales = read_sales('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
%%time

sub_valid = forecast(sales)
%%time

sales = read_sales('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
%%time

sub_eval = forecast(sales)
submission = pd.concat([sub_valid, sub_eval]).reset_index(drop=True)
submission.to_csv('submission.csv', index=False, float_format='%.5g')
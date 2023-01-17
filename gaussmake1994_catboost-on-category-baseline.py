%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dftrain = pd.read_csv('../input/train.tsv', delimiter='\t')

dftest = pd.read_csv('../input/test.tsv', delimiter='\t')
dftrain.head()
dftrain['category_name'] = dftrain['category_name'].fillna('NaN')

dftrain['brand_name'] = dftrain['brand_name'].fillna('NaN')

dftest['category_name'] = dftest['category_name'].fillna('NaN')

dftest['brand_name'] = dftest['brand_name'].fillna('NaN')
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, make_scorer

from math import sqrt





def RMSLE(y_true, y_pred, *args, **kwargs):

    return sqrt(mean_squared_error(np.log(y_true + 1e-5), np.log(y_pred + 1e-5), *args, **kwargs))





RMSLE_scorer = make_scorer(RMSLE)
from sklearn.dummy import DummyRegressor
model = DummyRegressor()

cross_val_score(model,

                np.zeros([len(dftrain), 1]),

                dftrain['price'],

                scoring=RMSLE_scorer)
from catboost import CatBoostRegressor
from tqdm import tqdm_notebook as tqdm





def category_split(category):

    parts = map(str.strip, category.split('/'))

    result = []

    previous = ""

    for part in parts:

        result.append("{0}_{1}".format(previous, part))

        previous = result[-1]

    return result
categories_train = [category_split(category) for category in tqdm(dftrain['category_name'])]

categories_test = [category_split(category) for category in tqdm(dftest['category_name'])]
max(map(len, categories_train))
def category_level(categories, level):

    result = []

    for category in tqdm(categories):

        if len(category) > level:

            result.append(category[level])

        else:

            result.append("")

    return result
for i in tqdm(range(5)):

    dftrain['category_name_{0}'.format(i)] = category_level(categories_train, i)

    dftest['category_name_{0}'.format(i)] = category_level(categories_test, i)
dftrain.head()
model = CatBoostRegressor(iterations=200, verbose=True)

cross_val_score(model,

                np.array(dftrain[['item_condition_id',

                                  'category_name_0', 

                                  'category_name_1',

                                  'category_name_2',

                                  'category_name_3',

                                  'category_name_4',

                                  'brand_name',

                                  'shipping']]),

                np.array(dftrain['price']),

                fit_params={

                    'cat_features': [1, 2, 3, 4, 5, 6]

                },

                scoring=RMSLE_scorer)
model = CatBoostRegressor(iterations=200, verbose=True)

model.fit(np.array(dftrain[['item_condition_id',

                            'category_name_0', 

                            'category_name_1',

                            'category_name_2',

                            'category_name_3',

                            'category_name_4',

                            'brand_name',

                            'shipping']]),

          np.array(dftrain['price']),

          cat_features=[1,2,3,4,5,6])
prediction = model.predict(np.array(dftest[['item_condition_id',

                                            'category_name_0', 

                                            'category_name_1',

                                            'category_name_2',

                                            'category_name_3',

                                            'category_name_4',

                                            'brand_name',

                                            'shipping']]))
from collections import OrderedDict



submission = pd.DataFrame(OrderedDict([

    ('test_id', dftest['test_id']),

    ('price', prediction)

]))

submission.head()
submission.to_csv('submission.csv', index=None)
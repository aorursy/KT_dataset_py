%matplotlib inline



import math



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
# vectorized error calc

def rmsle(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
data = pd.read_csv('../input/train.csv')

data.fillna(0, inplace=True)



# remove SalePrice outliers

data = data[np.abs(data.SalePrice-data.SalePrice.mean())<=(2.6*data.SalePrice.std())]
neigborhood_prices_per_sqmtr = dict()

for neighborhood in set(data.Neighborhood.values):

    rows = data[data['Neighborhood'] == neighborhood]

    mean_nb_price = np.mean(rows['SalePrice'] / (rows['GrLivArea'] * 0.093))

    neigborhood_prices_per_sqmtr[neighborhood] = int(mean_nb_price)



saletype_prices = dict()

for saletype in set(data.SaleType.values):

    rows = data[data['SaleType'] == saletype]

    mean_saletype_price = np.mean(rows['SalePrice'])

    saletype_prices[saletype] = int(mean_saletype_price)
def add_mean_nb_price(group):

    """

    Compute the mean price per square meter for each neighborhood

    and add this information into a specific feature

    """

    group['MeanNeighborhoodPricePerSqMtr'] = neigborhood_prices_per_sqmtr[group.Neighborhood]

    return group





def add_mean_saletype_price(group):

    """

    Compute the mean price per sale type

    and add this information into a specific feature

    """

    if group.SaleType in saletype_prices:

        group['MeanSaleTypePrice'] = saletype_prices[group.SaleType]

    else:

        group['MeanSaleTypePrice'] = np.mean(list(saletype_prices.values()))

    return group





def add_central_air(group):

    """

    Transforms CentralAir categorical feature into a binary numerical feature {0,1}

    """

    group['HasCentralAir'] = 1 if group['CentralAir'] == 'Y' else 0

    return group





features_straight = [

    'OverallQual',

    'YearBuilt',

    'GrLivArea',

    'MeanNeighborhoodPricePerSqMtr',

    'MeanSaleTypePrice',

    'HasCentralAir',

    'GarageArea',

    'FullBath',

    'Fireplaces',

    'LotFrontage',

]



feature_eng = [

    {'dest': 'AgeSold', 'func': lambda x: x['YrSold'] - x['YearBuilt']},

    {'dest': 'AgeSoldSquare', 'func': lambda x: (x['YrSold'] - x['YearBuilt']) ** 2},

    {'dest': 'YearBuiltSquare', 'func': lambda x: x['YearBuilt'] ** 2},

    {'dest': 'YearBuiltLog', 'func': lambda x: np.log(x['YearBuilt'])},

    {'dest': 'GrLivAreaSquare', 'func': lambda x: x['GrLivArea'] ** 2},

    {'dest': 'GrLivAreaCube', 'func': lambda x: x['GrLivArea'] ** 3},

    {'dest': 'GrLivAreaLog', 'func': lambda x: np.log(x['GrLivArea'])},

    {'dest': 'GarageAreaSquare', 'func': lambda x: x['GarageArea'] ** 2},

    {'dest': 'OverallQualSquare', 'func': lambda x: x['OverallQual'] ** 2},

    {'dest': 'OverallQualCube', 'func': lambda x: x['OverallQual'] ** 3},

]



removes = []
data = data.apply(add_mean_nb_price, axis=1)

data = data.apply(add_mean_saletype_price, axis=1)

data = data.apply(add_central_air, axis=1)

data_selected = data[features_straight]



for feature in feature_eng:

    data_selected[feature['dest']] = feature['func'](data)



for remove in removes:

    del data_selected[remove]



data_selected.fillna(0, inplace=True)



X, Y = data_selected, np.log10(data['SalePrice'])
cv = 100

rmsle_scores = []



for i in range(cv):

    X_train, X_test, Y_train, Y_test = train_test_split(data_selected, np.log10(data['SalePrice']), test_size=0.1)



    # regression on training subset

    forest = Pipeline([("random_forest", RandomForestRegressor(n_estimators=20))])

    line = Pipeline([("linear_regression", LinearRegression())])

    forest.fit(X_train, Y_train)

    line.fit(X_train, Y_train)



    # prediction on testing subset

    Y_test_predict = (forest.predict(X_test) + line.predict(X_test)) / 2

    Y_test_corrected = np.power(10, Y_test)

    Y_test_predict_corrected = np.power(10, Y_test_predict)

    

    rmsle_scores.append(rmsle(Y_test_corrected, Y_test_predict_corrected))



print('Mean RMSLE: ', np.mean(rmsle_scores))

print('Std RMSLE:', np.std(rmsle_scores))

plot = sns.distplot(rmsle_scores, bins=40)
# linear regression on the whole set

forest.fit(X, Y)

line.fit(X, Y)
submission = pd.read_csv('../input/test.csv')

submission = submission.apply(add_mean_nb_price, axis=1)

submission = submission.apply(add_mean_saletype_price, axis=1)

submission = submission.apply(add_central_air, axis=1)



submission_selected = submission[features_straight]



for feature in feature_eng:

    submission_selected[feature['dest']] = feature['func'](submission)



submission_selected.fillna(0, inplace=True)
prices = (forest.predict(submission_selected) + line.predict(submission_selected)) / 2

submission_to_csv = pd.DataFrame()

submission_to_csv['SalePrice'] = np.power(10, prices)

submission_to_csv.index = submission['Id']

print('Min:\t', submission_to_csv.SalePrice.min())

print('Max:\t', submission_to_csv.SalePrice.max())

print('Mean:\t', submission_to_csv.SalePrice.mean())

print('Std:\t', submission_to_csv.SalePrice.std())

plot = sns.distplot(submission_to_csv.SalePrice)
submission_to_csv.to_csv('submission.csv')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/train.csv')

df.head()
enumSeries = []

numSeries = []

nonDiscreteSeries = []

for colname in df.columns:

    datatype = str(df[colname].dtype)

    if colname == 'Id' or colname == 'SalePrice':

        pass

    elif datatype == 'int64' or datatype == 'float64':

        numSeries.append(colname)

    else:

        enumSeries.append(colname)

df.fillna(pd.Series([0,1970,0], index=['LotFrontage', 'GarageYrBlt','MasVnrArea']), inplace=True)

assert df[pd.isnull(df[numSeries]).any(axis=1)][numSeries].shape[0] == 0

df.head()
from scipy.stats import shapiro



totalCount = df.shape[0]

candidateForLog10 = []

for seriesName in numSeries:

    vcount = df[seriesName].value_counts().shape[0]

    zeros = np.sum((df[seriesName] == 0).values)

    if (vcount > totalCount / 10 and zeros == 0):

        shapLog10 = shapiro(np.log10(df[seriesName].values))[0]

        shapAsIs = shapiro(df[seriesName].values)[0]

        candidateForLog10.append((seriesName, shapLog10, shapAsIs, shapLog10 - shapAsIs))

candidateForLog10
df['LotAreaLog10'] = np.log10(df['LotArea'].values)

df['1stFlrSFLog10'] = np.log10(df['1stFlrSF'].values)

df['GrLivAreaLog10'] = np.log10(df['GrLivArea'].values)

df['SalePriceLog10'] = np.log10(df['SalePrice'].values)

if 'LotArea' in numSeries:

    numSeries.remove('LotArea')

    numSeries.append('LotAreaLog10')

if '1stFlrSF' in numSeries:

    numSeries.remove('1stFlrSF')

    numSeries.append('1stFlrSFLog10')

if 'GrLivArea' in numSeries:

    numSeries.remove('GrLivArea')

    numSeries.append('GrLivAreaLog10')

df[numSeries].head()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge



headdf = df.head(int(3.0 * df.shape[0] / 4.0) - 1)

taildf = df.tail(int(1.0 * df.shape[0] / 4.0))

print(str(headdf.shape[0]) + ' of the training set for training')



ridge = Ridge()

parameters = { 'alpha': [ 1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20 ]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(headdf[numSeries].values, headdf['SalePriceLog10'])

print(ridge_regressor.best_params_)
from sklearn.metrics import mean_absolute_error

ridge = Ridge(alpha=1)

ridge.fit(headdf[numSeries].values, headdf['SalePriceLog10'])

ridgePredictions = np.power(10, ridge.predict(taildf[numSeries].values))

targets = np.power(10, taildf['SalePriceLog10'])

mae = mean_absolute_error(ridgePredictions, targets)

mae
numSeries = abs(df[numSeries].corrwith(df['SalePriceLog10'])).sort_values(ascending=False).keys().values.tolist()

numSeries
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt



maeResults = []

maeNames = []

for endIndex in range(1, len(numSeries)):

    ridge = Ridge(alpha=1)

    ridge.fit(headdf[numSeries[0:endIndex]].values, headdf['SalePriceLog10'])

    ridgePredictions = np.power(10, ridge.predict(taildf[numSeries[0:endIndex]].values))

    targets = np.power(10, taildf['SalePriceLog10'])

    mae = mean_absolute_error(ridgePredictions, targets)

    maeResults.append(mae)

    if endIndex > 1:

        maeNames.append(str(endIndex) + ' characteristics')

    else:

        maeNames.append(str(endIndex) + ' characteristic')

maeResultsSeries = pd.Series(maeResults, index=maeNames)

plt.plot(range(1, maeResultsSeries.size + 1), maeResultsSeries, marker='o',);
queries = []

queries.append('MSZoning != "RM" and GarageFinish != "Fin" and ExterQual != "Gd"')

queries.append('MSZoning != "RM" and GarageFinish != "Fin" and ExterQual == "Gd"')

queries.append('MSZoning != "RM" and GarageFinish == "Fin" and ExterQual != "Gd"')

queries.append('MSZoning != "RM" and GarageFinish == "Fin" and ExterQual == "Gd"')

queries.append('MSZoning == "RM"')

maeResults = []

maeNames = []

ridgePredictions = []

targets = []

for query in queries:

    ridge = Ridge(alpha=1)

    ridge.fit(headdf.query(query)[numSeries].values, headdf.query(query)['SalePriceLog10'])

    preds = np.power(10, ridge.predict(taildf.query(query)[numSeries].values))

    ridgePredictions.extend(preds)

    targets.extend(np.power(10, taildf.query(query)['SalePriceLog10']))

mae = mean_absolute_error(ridgePredictions, targets)

mae
from sklearn.linear_model import Lasso



lasso = Lasso()

parameters = { 'alpha': [ 0.01, 0.1, 1, 5, 10, 20 ]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

lasso_regressor.fit(headdf[numSeries].values, headdf['SalePriceLog10'])

print(lasso_regressor.best_params_)
maeResults = []

maeNames = []

lassoPredictions = []

targets = []

for query in queries:

    lasso = Lasso(alpha=0.01)

    lasso.fit(headdf.query(query)[numSeries].values, headdf.query(query)['SalePriceLog10'])

    preds = np.power(10, lasso.predict(taildf.query(query)[numSeries].values))

    lassoPredictions.extend(preds)

    targets.extend(np.power(10, taildf.query(query)['SalePriceLog10']))

mae = mean_absolute_error(lassoPredictions, targets)

mae
tdf = pd.read_csv('../input/test.csv')

tdf.fillna(pd.Series([0,1970,0, 5, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0], index=['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'OverallQual', 'GrLivArea10', 'GarageCars', '1stFlrSF10', 'TotalBsmtSF', 'GarageArea', 'BsmtFullBath', 'BsmtUnfSF', 'BsmtHalfBath', 'BsmtFinSF2', 'BsmtFinSF1']), inplace=True)

tdf['LotAreaLog10'] = np.log10(tdf['LotArea'].values)

tdf['1stFlrSFLog10'] = np.log10(tdf['1stFlrSF'].values)

tdf['GrLivAreaLog10'] = np.log10(tdf['GrLivArea'].values)

assert tdf[pd.isnull(tdf[numSeries]).any(axis=1)][numSeries].shape[0] == 0

maeResults = []

maeNames = []

ridgePredictions = []

ridgePredictionIds = []

for query in queries:

    ridge = Ridge(alpha=1)

    ridge.fit(df.query(query)[numSeries].values, df.query(query)['SalePriceLog10'])

    preds = np.power(10, ridge.predict(tdf.query(query)[numSeries].values))

    ridgePredictions.extend(preds)

    ridgePredictionIds.extend(tdf.query(query)['Id'].values)

res = pd.DataFrame([ridgePredictionIds, ridgePredictions]).transpose().rename(columns={0: 'Id', 1: 'SalePrice'}).sort_values(by=['Id'])

res.Id = res.Id.astype(int)

res.to_csv('submission.csv', index=False)

res
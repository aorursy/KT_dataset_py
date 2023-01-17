import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/train.csv')

target = pd.read_csv('../input/test.csv')



# replace prices with logarithmic prices

data['logSalePrice'] = np.log10(data.SalePrice.values)

data.drop('SalePrice', axis=1, inplace=True)
print(open('../input/data_description.txt', 'r').read())
continuous_features = [col for col in data.columns if data[col].dtype != 'O']



continuous_features.remove('Id')

continuous_features.remove('logSalePrice') # remove the target feature



ordinal_features = ['Street', 'Alley', 'LotShape', 'Utilities', 'LandSlope', 'ExterQual', 

                    'LandContour', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 

                    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 

                    'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType',

                    'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 

                    'Fence'] 



categorical_features = ['LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 

                        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 

                        'MasVnrType', 'Foundation', 'Heating', 'MiscFeature',

                        'SaleType', 'SaleCondition', 'MSZoning', 'BldgType']



(len(set(continuous_features + ordinal_features + categorical_features)) == 

 len(continuous_features + ordinal_features + categorical_features) == 

 len(data.columns)-2)
print('incomplete features in data:\n' +

      "\n".join(["  {}: {} useful data points".format(col, data[col].dropna().count()) 

                 for col in data.columns if data[col].dropna().count() < 1460]), '\n')

print('incomplete features in target:\n' +

      "\n".join(["  {}: {} useful data points".format(col, target[col].dropna().count()) 

                 for col in target.columns if target[col].dropna().count() < 1459]))
from scipy.stats import linregress



# linear regression on LotFrontage and sqrt(LotArea); cutoff at LotArea<20000 to minimize outliers

slope, intercept, rvalue, pvalue, stderr = linregress(

    np.sqrt(data.loc[(data.LotFrontage.notna()) & (data.LotArea < 20000), 'LotArea']),

    data.loc[(data.LotFrontage.notna()) & (data.LotArea < 20000), 'LotFrontage'])



# derive residuals for known LotFrontage

residual = data.LotFrontage - (slope*np.sqrt(data.LotArea)+intercept)

print('mean residual:', np.nanmean(residual), '\nmedian residual:', 

      np.nanmedian(residual), '\nresidual std:', np.nanstd(residual))



# apply equation for missing data

data.loc[data.LotFrontage.isna(), 'LotFrontage'] = np.sqrt(data.LotArea)*slope+intercept

target.loc[target.LotFrontage.isna(), 'LotFrontage'] = np.sqrt(target.LotArea)*slope+intercept
# fill nan with 'NA' for further processing (see below)

for col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 

            'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 

            'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:

    data.loc[:, col].fillna('NA', inplace=True)

    target.loc[:, col].fillna('NA', inplace=True)    
for sample in [data, target]:

    sample.fillna({col: sample.loc[:, col].median() for col in continuous_features}, inplace=True)

    sample.fillna({col: sample.loc[:, col].value_counts().index[0] for col in categorical_features + ordinal_features}, 

                  inplace=True)
print('incomplete features in data:', 

      len(["  {}: {} useful data points".format(col, data[col].dropna().count()) 

           for col in data.columns if data[col].dropna().count() < 1460]), '\n')

print('incomplete features in target:', 

      len(["  {}: {} useful data points".format(col, target[col].dropna().count()) 

           for col in target.columns if target[col].dropna().count() < 1459]))
ordinal_transform = {'Street':  {'Grvl': 1, 'Pave': 2}, 

                     'Alley': {'NA': 0, 'Grvl': 1, 'Pave': 2}, 

                     'LotShape': {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}, 

                     'Utilities': {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4}, 

                     'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3}, 

                     'LandContour': {'Low': 1, 'HLS': 1, 'Bnk': 2, 'Lvl': 3}, 

                     'ExterQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                     'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'BsmtQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'BsmtCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'BsmtExposure': {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}, 

                     'BsmtFinType1': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 

                                      'GLQ': 6}, 

                     'BsmtFinType2': {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 

                                      'GLQ': 6}, 

                     'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},

                     'CentralAir': {'N': 1, 'Y': 2}, 

                     'Electrical': {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5}, 

                     'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'Functional': {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 

                                    'Min1': 7, 'Typ': 8}, 

                     'FireplaceQu': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'GarageType': {'NA': 0, 'Detchd': 1, 'CarPort': 2, 'BuiltIn': 3, 

                                    'Basment': 4, 'Attchd': 5, '2Types': 6},

                     'GarageFinish': {'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}, 

                     'GarageQual': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'GarageCond': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'PavedDrive': {'N': 1, 'P': 2, 'Y': 3}, 

                     'PoolQC': {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 

                     'Fence': {'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}, 

                    }



# apply transformations

for col in ordinal_features:

    data.loc[:, col] = data.loc[:, col].map(ordinal_transform[col], na_action='ignore')

    target.loc[:, col] = target.loc[:, col].map(ordinal_transform[col], na_action='ignore')

    

# move some features from continuous to ordinal feature list

for col in ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 

            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']:

    continuous_features.remove(col)

    ordinal_features.append(col)

    

# move one feature from continuous to categorial feature list

continuous_features.remove('MSSubClass')

categorical_features.append('MSSubClass')
from sklearn.preprocessing import MultiLabelBinarizer



enc = MultiLabelBinarizer()

enc.fit([data.Condition1, data.Condition2])



# apply transformation to data sample

binarized_columns = pd.DataFrame(enc.transform(list(zip(data.Condition1, data.Condition2))), 

                                 columns=enc.classes_, index=data.Id)

data = pd.merge(data, binarized_columns, on='Id')

data.drop(['Condition1', 'Condition2'], axis=1, inplace=True)



# apply transformation to target sample

binarized_columns = pd.DataFrame(enc.transform(list(zip(target.Condition1, target.Condition2))), 

                                 columns=enc.classes_, index=target.Id)

target = pd.merge(target, binarized_columns, on='Id')

target.drop(['Condition1', 'Condition2'], axis=1, inplace=True)



categorical_features.remove('Condition1')

categorical_features.remove('Condition2')
from sklearn.preprocessing import LabelBinarizer



enc = LabelBinarizer()



ordinalized = []

binary_features = []

binarize = ['Neighborhood', 'MSSubClass']



for col in categorical_features:

    if col not in binarize:

        # rank by median logSalePrice

        sorted_labels = data[

            [col, 'logSalePrice']].groupby(col).logSalePrice.median().sort_values()

        data.loc[:, col] = data.loc[:, col].map(

            {sorted_labels.index.values[i]:i for i in range(len(sorted_labels))})

        target.loc[:, col] = target.loc[:, col].map(

            {sorted_labels.index.values[i]:i for i in range(len(sorted_labels))})

        ordinalized.append(col)

    else:

        # binarize

        enc.fit(data[col])

        binarized_columns = pd.DataFrame(enc.transform(data[col]), 

                                         columns=enc.classes_, index=data.Id)

        data = pd.merge(data, binarized_columns, on='Id')

        data.drop(col, axis=1, inplace=True)

        binarized_columns = pd.DataFrame(enc.transform(target[col]), 

                                         columns=enc.classes_, index=target.Id)

        target = pd.merge(target, binarized_columns, on='Id')

        target.drop(col, axis=1, inplace=True)

        binary_features += list(enc.classes_)



for col in binarize:        

    categorical_features.remove(col)

    

for col in ordinalized:

    categorical_features.remove(col)

    ordinal_features.append(col)
f, ax = plt.subplots(int(np.floor(len(ordinal_features)/5)+1), 5, 

                     figsize=(15, (np.floor(len(ordinal_features)/5)+1)*3),

                     sharey=True)

ax = np.ravel(ax)

plt.subplots_adjust(hspace=0.5)



for i in range(len(ordinal_features)):

    data.boxplot(column='logSalePrice', by=ordinal_features[i], ax=ax[i])
f, ax = plt.subplots(int(np.floor(len(continuous_features)/5))+1, 5, 

                     figsize=(15, int(np.floor(len(continuous_features)/5)+1)*3),

                     sharey=True)

ax = np.ravel(ax)



plt.subplots_adjust(hspace=0.3)



for i in range(len(continuous_features)):

    data.plot.scatter(x=continuous_features[i], y='logSalePrice',

                      alpha=0.1, legend=True, s=5, ax=ax[i])
data['totalyear'] = np.sqrt(4020.1-data.YearBuilt-data.YearRemodAdd)

target['totalyear'] = np.sqrt(4020.1-target.YearBuilt-target.YearRemodAdd)

data.plot.scatter(x='totalyear', y='logSalePrice', alpha=0.1)
data['basement'] = (data.BsmtFinType1 * data.BsmtFinSF1 + 

                    data.BsmtFinType2 * data.BsmtFinSF2)

target['basement'] = (target.BsmtFinType1 * target.BsmtFinSF1 + 

                      target.BsmtFinType2 * target.BsmtFinSF2)

data.drop(['BsmtFinType1', 'BsmtFinType2', 

           'BsmtFinSF1', 'BsmtFinSF2'], axis=1, inplace=True)

plt.scatter(data.basement, data.logSalePrice, alpha=0.1)
data['exterior'] = data.Exterior1st + data.Exterior2nd

target['exterior'] = target.Exterior1st + target.Exterior2nd

data.drop(['Exterior1st', 'Exterior2nd'], axis=1, inplace=True)

plt.scatter(data.exterior, data.logSalePrice, alpha=0.1)
data['squarefeet'] = (data.TotalBsmtSF + 

                      data['1stFlrSF'] + data['2ndFlrSF'])

target['squarefeet'] = (target.TotalBsmtSF + target['1stFlrSF'] + 

                        target['2ndFlrSF'])

plt.scatter(data.squarefeet, data.logSalePrice, alpha=0.1)
data['baths'] = (3*data.FullBath + data.HalfBath + 

                 2*data.BsmtFullBath + data.BsmtHalfBath)

target['baths'] = (3*target.FullBath + target.HalfBath + 

                   2*target.BsmtFullBath + target.BsmtHalfBath)

plt.scatter(data.baths, data.logSalePrice, alpha=0.1)
data['masonry'] = np.sqrt(data.MasVnrArea) * data.MasVnrType

target['masonry'] = np.sqrt(target.MasVnrArea) * target.MasVnrType

data.drop(['MasVnrArea', 'MasVnrType'], axis=1, inplace=True)

plt.scatter(data.masonry, data.logSalePrice, alpha=0.1)
data['kitchen'] = data.KitchenAbvGr+data.KitchenQual

target['kitchen'] = target.KitchenAbvGr+target.KitchenQual

plt.scatter(data.kitchen, data.logSalePrice, alpha=0.1)
data['fireplace'] = np.sqrt(data.Fireplaces*data.FireplaceQu)

target['fireplace'] = np.sqrt(target.Fireplaces*target.FireplaceQu)

plt.scatter(data.fireplace, data.logSalePrice, alpha=0.1)
data['kitchenbaths'] = data.kitchen+data.baths

target['kitchenbaths'] = target.kitchen+target.baths

plt.scatter(data.kitchenbaths, data.logSalePrice, alpha=0.1)
data['garage'] = data.GarageArea*(data.GarageQual+data.GarageCond+

                                  2*data.GarageCars+2*data.GarageFinish)

target['garage'] = target.GarageArea*(target.GarageQual+target.GarageCond+

                                      2*target.GarageCars+2*target.GarageFinish)

plt.scatter(data.garage, data.logSalePrice, alpha=0.1)
data['outdoorarea'] = (data.WoodDeckSF + data.OpenPorchSF + data.EnclosedPorch + 

                       data['3SsnPorch'] + data.ScreenPorch + data.PoolArea)

target['outdoorarea'] = (target.WoodDeckSF + target.OpenPorchSF + target.EnclosedPorch + 

                         target['3SsnPorch'] + target.ScreenPorch + target.PoolArea)

data.plot.scatter(x='outdoorarea', y='logSalePrice', alpha=0.1)
data['outdoor'] = 0

target['outdoor'] = 0

for col in ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

            'ScreenPorch', 'PoolArea']:

    data.loc[data[col] > 0, 'outdoor'] = 1

    target.loc[target[col] > 0, 'outdoor'] = 1

    

data.groupby('outdoor').logSalePrice.plot.hist(alpha=0.5)
data['totalquality'] = (data.OverallQual + data.KitchenQual + data.BsmtQual + 

                        data.ExterQual + data.HeatingQC + data.FireplaceQu + 

                        data.GarageQual)

target['totalquality'] = (target.OverallQual + target.KitchenQual + target.BsmtQual + 

                          target.ExterQual + target.HeatingQC + target.FireplaceQu + 

                          target.GarageQual)

plt.scatter(data.totalquality, data.logSalePrice, alpha=0.1)
data['totalcond'] = (data.OverallCond + data.BsmtCond + data.ExterCond + 

                     data.GarageCond)

target['totalcond'] = (target.OverallCond + target.BsmtCond + target.ExterCond + 

                       target.GarageCond)

plt.scatter(data.totalcond, data.logSalePrice, alpha=0.1)
f, ax = plt.subplots(figsize=(20,6))



data.drop('logSalePrice', axis=1).corrwith(

    data.logSalePrice).agg('square').plot.bar(ax=ax, alpha=0.5)

ax.plot(ax.get_xlim(), [0.6, 0.6], color='red')

ax.set_title('Coefficient of Determination')

ax.grid()
from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=2, include_bias=True)



poldata = poly.fit_transform(data[['squarefeet', 'kitchenbaths', 

                                   'totalquality']]).transpose()

for colid in range(poldata.shape[0]):

    data['poly_{}'.format(colid)] = poldata[colid]

    

poldata = poly.fit_transform(target[['squarefeet', 'kitchenbaths', 

                                     'totalquality']]).transpose()

for colid in range(poldata.shape[0]):

    target['poly_{}'.format(colid)] = poldata[colid]
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge



featurelist = data.drop('logSalePrice', axis=1).columns



pipe = Pipeline([('scaler', RobustScaler()), 

                 ('model', Ridge(fit_intercept=True, random_state=42))])



param_space = {'model__alpha': [1, 5, 10]}



grid = GridSearchCV(pipe, param_grid=param_space, cv=10, 

                    scoring='neg_mean_squared_error')



grid.fit(data[featurelist], data.logSalePrice)
grid.best_params_
10**np.sqrt(-grid.best_score_)
pred = pd.DataFrame({'SalePrice': 10**grid.predict(target[featurelist])}, index=target.Id)
pred.to_csv('prediction.csv', index='Id')
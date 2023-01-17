# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train  = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import  SVR

from sklearn.model_selection import cross_val_score
def do_cross_validation(X_train, y_train):

    models = []

    models.append(('LR', LinearRegression()))

    models.append(('LASSO', Lasso()))

    models.append(('EN', ElasticNet()))

    models.append(('Ridge', Ridge()))

    models.append(('CART', DecisionTreeRegressor()))

    models.append(('KNN', KNeighborsRegressor()))

    models.append(('SVR', SVR()))

    results = []

    names = []

    scoring = 'neg_mean_squared_error'

    for name, model in models:

        cv_results = np.sqrt(-cross_val_score(model, X_train, y_train, cv=10, scoring=scoring))

        results.append(cv_results)

        names.append(name)

        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
X_train = train[['YrSold', 'MoSold', 'LotArea', 'BedroomAbvGr']]

y_train = np.log(train['SalePrice'])

X_test = test[['YrSold', 'MoSold', 'LotArea', 'BedroomAbvGr']]

do_cross_validation(X_train, y_train)
model = LinearRegression()

model.fit(X_train, y_train)

pred = np.exp(model.predict(X_test))

submission = pd.DataFrame(data={'Id':test.Id, 'SalePrice': pred})

submission.to_csv('Benchmark.csv', index=False)
all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],

                     test.loc[:,'MSSubClass':'SaleCondition']))
# Get all the numeric features

numeric_cols = all_data.dtypes[all_data.dtypes !='object'].index

numeric_data = all_data[numeric_cols]

print(numeric_data.shape)
# missing values

numeric_data = numeric_data.fillna(numeric_data.mean())
X_train = numeric_data[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = numeric_data[train.shape[0]:]

do_cross_validation(X_train, y_train)
model = Ridge()

model.fit(X_train, y_train)

pred = np.exp(model.predict(X_test))

submission = pd.DataFrame(data={'Id':test.Id, 'SalePrice': pred})

submission.to_csv('Ridge_FeatureProcessing1.csv', index=False)
from scipy.stats import skew

skewed_cols = numeric_data.apply(lambda x: skew(x))

skewed_cols = skewed_cols[skewed_cols > 0.75].index

numeric_data[skewed_cols] = np.log1p(numeric_data[skewed_cols])
X_train = numeric_data[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = numeric_data[train.shape[0]:]

do_cross_validation(X_train, y_train)
model = Ridge()

model.fit(X_train, y_train)

pred = np.exp(model.predict(X_test))

submission = pd.DataFrame(data={'Id':test.Id, 'SalePrice': pred})

submission.to_csv('Ridge_FeatureProcessing2.csv', index=False)
categori_cols = all_data.dtypes[all_data.dtypes == 'object'].index

categori_data = all_data[categori_cols]

categori_data = pd.get_dummies(categori_data)

print(categori_data.shape)

combined_data = pd.concat([numeric_data, categori_data], axis=1)

print(combined_data.shape)
X_train = combined_data[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = combined_data[train.shape[0]:]

do_cross_validation(X_train, y_train)
model = Ridge()

model.fit(X_train, y_train)

pred = np.exp(model.predict(X_test))

submission = pd.DataFrame(data={'Id':test.Id, 'SalePrice': pred})

submission.to_csv('Ridge_FeatureProcessing3.csv', index=False)
all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],

                     test.loc[:,'MSSubClass':'SaleCondition']))

all_data.columns
all_data.shape
#check for NaN

all_data.MSSubClass.isnull().sum()
all_data.MSSubClass.value_counts()
# MSSubClass - could be more useful as a categorical variable

all_data_new = pd.get_dummies(all_data.MSSubClass.replace([190, 85, 75, 45, 180, 40, 150], 'Other'), prefix='MSSubClass')

all_data_new.shape
all_data_new.columns
# check for NaN

all_data.MSZoning.value_counts()
all_data_new.shape


new_var = pd.get_dummies(all_data.MSZoning, prefix='MSZoning')

print(all_data_new.shape)

print(new_var.shape)

all_data_new = pd.concat([all_data_new, new_var], axis=1)

print(all_data_new.shape)
all_data.LotFrontage.isnull().sum()
new_var = all_data.LotFrontage.fillna(all_data.LotFrontage.mean())

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data.LotArea.isnull().sum()
new_var = np.log(all_data['LotArea'])

all_data_new  = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data.Street.value_counts()
new_var = pd.get_dummies(all_data.Street, prefix='Street')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data.Alley.isnull().sum()
all_data.Alley.value_counts()
new_var = pd.get_dummies(all_data.Alley.fillna('None'), prefix='Alley')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data.LotShape.isnull().sum()
all_data.LotShape.value_counts()
train.groupby('LotShape')['SalePrice'].mean()
all_data_new['LotShape'] = np.where(all_data.LotShape == 'Reg', 1, 0)

all_data_new.shape
all_data.LandContour.isnull().sum()
all_data.LandContour.value_counts()
new_var = pd.get_dummies(all_data.LandContour, prefix='LandContour')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data.Utilities.value_counts()
all_data.LotConfig.isnull().sum()
new_var = pd.get_dummies(all_data.LotConfig, prefix='LotConfig')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data.LandSlope.isnull().sum()
all_data.LandSlope.value_counts()
new_var = pd.get_dummies(all_data.LandSlope, prefix='LandSlope')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data.Neighborhood.isnull().sum()
new_var = pd.get_dummies(all_data.Neighborhood, 'Neighborhood')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new.shape
new_var = pd.get_dummies(all_data.Condition1, 'Condition1')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
new_var = pd.get_dummies(all_data.BldgType, 'BldgType')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.HouseStyle, 'HouseStyle')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new = pd.concat([all_data_new, all_data.OverallQual], axis=1)

all_data_new.shape
all_data_new = pd.concat([all_data_new, all_data.OverallCond], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
year_built_bins = pd.qcut(all_data['YearBuilt'], 5, labels=['oldest', 'old', 'middle', 'new', 'newest'])

new_var = pd.get_dummies(year_built_bins, 'YearBuilt')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
# skip YearRemodAdd
temp = all_data['RoofStyle'].replace(['Gambrel', 'Flat', 'Mansard', 'Shed'], 'Other')

new_var = pd.get_dummies(temp, 'RoofStyle')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new['RoofMatl'] = (all_data['RoofMatl'] == 'CompShg') * 1

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
temp = all_data['Exterior1st'].replace(['BrkComm', 'AsphShn', 'Stone', 'CBlock', 'ImStucc'], 'Other')

temp = temp.fillna('Other')

new_var = pd.get_dummies(temp, 'Exterior1st')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
temp = all_data['Exterior2nd'].replace(['Brk Cmn', 'AsphShn', 'Stone', 'CBlock', 'ImStucc'], 'Other')

temp = temp.fillna('Other')

new_var = pd.get_dummies(temp, 'Exterior2nd')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
new_var = pd.get_dummies(all_data.MasVnrType.fillna('None'), 'MasVnrType')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new['MasVnrArea'] = np.log(all_data['MasVnrArea'].fillna(0) + 1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.ExterQual, 'ExterQual')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
new_var = all_data['ExterCond'].map({'TA':0, 'Gd':1, 'Ex':1, 'Po':-1, 'Fa':-1})

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
temp = all_data.Foundation.replace(['Slab', 'Stone', 'Wood'], 'Other')

new_var = pd.get_dummies(temp, 'Foundation')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.BsmtQual.fillna('other'), 'BsmtQual')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.BsmtCond.fillna('None').replace(['Po'], 'Fa'), 'BsmtCond')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.BsmtExposure.fillna('No'), 'BsmtExposure')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
print(all_data.BsmtExposure.isnull().sum())

print(all_data.BsmtExposure.value_counts())

print(train.groupby('BsmtExposure')['SalePrice'].mean())
all_data.BsmtFinType2.value_counts()
new_var = pd.get_dummies(all_data.BsmtFinType1.fillna('Unf'), 'BsmtFinType1')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new['BsmtFinSF1'] = np.log(all_data.BsmtFinSF1.fillna(0) + 1)
new_var = pd.get_dummies(all_data.BsmtFinType2.fillna('Unf'), 'BsmtFinType2')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new['BsmtFinSF2'] = np.log(all_data.BsmtFinSF2.fillna(0) + 1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
all_data_new['BsmtUnfSF'] = np.log(all_data.BsmtUnfSF.fillna(0) + 1)

all_data_new.shape
all_data_new['TotalBsmtSF'] = np.log(all_data.TotalBsmtSF.fillna(0) + 1)

all_data_new.shape
temp = all_data.Heating.replace(['Grav', 'Wall', 'OthW', 'Floor'], 'Other')

new_var = pd.get_dummies(temp, 'Heating')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
new_var = pd.get_dummies(all_data.HeatingQC, 'HeatingQC')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new['CentralAir'] = np.where(all_data.CentralAir == 'Y', 1, 0)

all_data_new.shape
print(all_data.Electrical.isnull().sum())

print(all_data['Electrical'].value_counts())

print(train.groupby('Electrical')['SalePrice'].mean())
new_var = pd.get_dummies(all_data.Electrical.fillna('SBrkr'), 'Electrical')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new['1stFlrSF'] = np.log(all_data['1stFlrSF'])

all_data_new.shape
all_data_new['2ndFlrSF'] = np.log(all_data['2ndFlrSF'] + 1)
#ignored
all_data_new['GrLivArea'] = np.log(all_data.GrLivArea + 1)
new_var = pd.get_dummies(all_data.BsmtFullBath.fillna(0).replace(3, 2).astype(int), 'BsmtFullBath')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
# ignore

#all_data_new['BsmtHalfBath'] = all_data.BsmtHalfBath.fillna(0).replace(2, 1)

#print(all_data.BsmtHalfBath.value_counts())

#print(train.groupby('BsmtHalfBath')['SalePrice'].mean())
print(all_data.FullBath.value_counts())

print(train.groupby('FullBath')['SalePrice'].mean())
new_var = pd.get_dummies(all_data.FullBath.replace(0,2).astype(int), 'FullBath')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
new_var = pd.get_dummies(all_data.HalfBath, 'HalfBath')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
new_var = pd.get_dummies(all_data.BedroomAbvGr.replace([0,5,6,8,],1).astype(int), 'BedroomAbvGr')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new['KitchenAbvGr'] = (all_data.KitchenAbvGr >=2 ) * 1
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
all_data.KitchenQual.isnull().sum()
new_var = pd.get_dummies(all_data.KitchenQual.fillna('TA'), 'KitchenQual')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
temp = all_data['TotRmsAbvGrd'].map(lambda x: 0 if x < 4 else 1 if x < 7 else 2 if x < 11 else 3)

temp.shape
new_var = pd.get_dummies(temp, 'TotRmsAbvGrd')

all_data_new = pd.concat([all_data_new, new_var], axis = 1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.Functional.fillna('Maj1'), 'Functional')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.Fireplaces.replace([3,4], 2), 'Fireplaces')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.FireplaceQu.fillna('None'), 'FireplaceQu')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.GarageType.fillna('None'), 'GarageType')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
garage_year = pd.qcut(all_data.GarageYrBlt, 5, labels=['oldest', 'old', 'middle', 'new', 'newest'])

new_var = pd.get_dummies(garage_year.astype(object).fillna('None'), 'GarageYrBlt')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
all_data_new['GarageCars'] = all_data.GarageCars.fillna(0)
all_data_new['GarageArea'] = np.log(all_data.GarageArea.fillna(0) + 1)

all_data_new.shape
new_var = pd.get_dummies(all_data.GarageQual.fillna('None'), 'GarageQual')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
new_var = pd.get_dummies(all_data.GarageCond.fillna('None'), 'GarageCond')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
new_var = pd.get_dummies(all_data.PavedDrive, 'PavedDrive')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
all_data_new['WoodDeckSF'] = np.log(all_data.WoodDeckSF + 1)
all_data_new['OpenPorchSF'] = np.log(all_data.OpenPorchSF + 1)
all_data_new['EnclosedPorch'] = np.log(all_data.EnclosedPorch + 1)
all_data_new['3SsnPorch'] = np.log(all_data['3SsnPorch'] + 1)
all_data_new['ScreenPorch'] = np.log(all_data.ScreenPorch + 1)
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
#temp = all_data['Fence'].fillna('None').replace('MnWw', 'MnPrv')

#new_var = pd.get_dummies(temp, 'Fence')

#all_data_new = pd.concat([all_data_new, new_var], axis=1)

#all_data_new.shape
#all_data_new['MiscFeature'] = (all_data['MiscFeature'] == 'Shed') * 1
#all_data_new['MiscVal'] = np.log(all_data['MiscVal'] + 1)
temp = all_data['MoSold'].map({1:'Winter', 2:'Winter', 12:'Winter',

                          3:'Spring',4:'Spring',5:'Spring',

                          6:'Summer',7:'Summer',8:'Summer',

                          9:'Fall',10:'Fall',11:'Fall'})



new_var = pd.get_dummies(temp, 'MoSold')

all_data_new = pd.concat([all_data_new, new_var], axis=1)

all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
#new_var = pd.get_dummies(all_data['SaleType'].fillna('WD'), "SaleType")

#all_data_new = pd.concat([all_data_new, new_var], axis=1)

#all_data_new.shape
X_train = all_data_new[:train.shape[0]]

y_train = np.log(train['SalePrice'])

X_test = all_data_new[train.shape[0]:]

do_cross_validation(X_train, y_train)
#new_var = pd.get_dummies(all_data['SaleCondition'], 'SaleCondition')

#all_data_new = pd.concat([all_data_new, new_var], axis=1)

#all_data_new.shape
alphas = np.logspace(-2, 1.3, 100)

alpha_scores = {}

for alpha in alphas:

    test_scores = []

    model = Ridge(alpha=alpha)

    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    #score = np.sqrt(np.mean((y_pred_test - y_train) ** 2))

    score = np.sqrt(-cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))

    #test_scores.append(score)

    alpha_scores[alpha] = np.mean(score)

    

%matplotlib inline

pd.Series(alpha_scores).plot(); 
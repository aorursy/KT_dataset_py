#Import Packages



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer

from scipy import stats

from scipy.stats import skew, norm



#Read Data

train_ = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#GrLivArea vs. SalePrice

plt.scatter(train_['GrLivArea'], train_['SalePrice'])

plt.xlabel('Sq. Ft.')

plt.ylabel('Sales Price ($)')

plt.title('Sq. Ft. vs. Sales Price of Home in Ames, Iowa')
#remove high leverage points

train_ = train_[train_['GrLivArea'] < 4000]

#Combine Datasets

data = pd.concat((train_.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))
#Plot SalePrice against Normal Distribution

sns.distplot(train_['SalePrice'], fit=norm);

plt.show()

resid = stats.probplot(train_['SalePrice'], plot=plt)
#log transform SalePrice and replot

train = train_

train['SalePrice'] = np.log1p(train['SalePrice'])

#Pull out target variable

y = train['SalePrice']

#Replot data

sns.distplot(train['SalePrice'], fit=norm);

plt.show()

log_resid = stats.probplot(train['SalePrice'], plot=plt)
#Change Categorical to Numeric

data = data.replace({'Street' : {'Grvl' : 1, 'Pave' : 2},

                       'Alley' : {'Grvl' : 1, 'Pave' : 2},

                       'LotShape' : {'IR3' : 1, 'IR2' : 1, 'IR1' : 2, 'Reg' : 3},

                       'LandCountour': {'Low' : 1, 'HLS' : 2, 'Bnk' : 2, 'Lvl' : 3},

                       'Utilities' : {'ELO' : 1, 'NoSeWa' : 2, 'NoSewr' : 2, 'AllPub' : 3},

                       'LandSlope' : {'Sev' : 1, 'Mod' : 2, 'Gtl' : 3},

                       'BsmtExposure' : {'No' : 1, 'Mn' : 2, 'Av': 3, 'Gd' : 4},

                       'BsmtFinType1' : {'Unf' : 1, 'LwQ': 2, 'Rec' : 3, 'BLQ' : 4, 

                                         'ALQ' : 5, 'GLQ' : 6},

                       'BsmtFinType2' : {'Unf' : 1, 'LwQ': 2, 'Rec' : 3, 'BLQ' : 4, 

                                         'ALQ' : 5, 'GLQ' : 6},

                       'Functional' : {'Sal' : 1, 'Sev' : 2, 'Maj2' : 3, 'Maj1' : 4, 'Mod': 5, 

                                       'Min2' : 6, 'Min1' : 7, 'Typ' : 8},

                       'GarageFinish' : {'Unf' : 1, 'RFn' : 2, 'Fin' : 3},

                       'PavedDrive' : {'N' : 1, 'P' : 2, 'Y' : 3},

                       'PoolQC' : {'Fa' : 1, 'TA' : 2, 'Gd' : 3, 'Ex' : 4},

                       'Fence' : {'GdPrv' : 2, 'MnPrv' : 1, 'GdWo' : 2, 'MnWw' : 1},

                       'MiscFeature' : {'Elev' : 1, 'Gar2' : 1, 'Othr' : 1, 'Shed' : 1, 'TenC' : 1},

                       'SaleCondition' : {'Normal' : 2, 'Abnorml' : 1, 'AdjLand' : 1, 'Alloca' : 1, 'Family' : 1, 'Partial' : 1}

                      })

  

quality = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

    

data["ExterQual"] = data["ExterQual"].map(quality)

data["ExterCond"] = data["ExterCond"].map(quality)

data["BsmtQual"] = data["BsmtQual"].map(quality)

data["BsmtCond"] = data["BsmtCond"].map(quality)

data["HeatingQC"] = data["HeatingQC"].map(quality)

data["KitchenQual"] = data["KitchenQual"].map(quality)

data["FireplaceQu"] = data["FireplaceQu"].map(quality)

data["GarageQual"] = data["GarageQual"].map(quality)

data["GarageCond"] = data["GarageCond"].map(quality)



#Observe SalePrice by Neighborhood and group this column into numeric

train_.groupby('Neighborhood')['SalePrice'].mean().sort_values()

#Map Neighborhoods

neighbor_map = {

    'MeadowV':0, 

    'IDOTRR': 0,

    'BrDale': 0,

    'BrkSide':1,

    'Edwards':1,

    'OldTown':1,

    'Sawyer':1,

    'Blueste':1,

    'SWISU':1,

    'NPkVill':1,

    'NAmes':1,

    'Mitchel':1,

    'SawyerW':2,

    'NWAmes':2,

    'Gilbert':2,

    'Blmngtn':2,

    'CollgCr':2,

    'Crawfor':3,

    'ClearCr':3,

    'Somerst':3,

    'Veenker':3,

    'Timber':3,

    'StoneBr':4,

    'NoRidge':4,

    'NridgHt':4

}



data['Neighborhood'] = data['Neighborhood'].map(neighbor_map).astype('int')

    

#Change Numeric to Categorical

data = data.replace({'MSSubClass' : {20 : '1-Story, New', 30 : '1-Story, Old', 40 : '1-Story, w/ Attic', 45 : '1 1/2 Story, Unfinished', 

                                       50 : '1 1/2 Story, Finished', 60 : '2-Story, New', 70 : '2-Story, Old', 75 : '2 1/2 Story', 

                                       80 : 'Split', 85 : 'Split Foyer', 90 : 'Duplex', 120 : '1-Story, PUD', 

                                       150 : '1 1/2 Story, PUD', 160 : '2-Story, PUD', 180 : 'Multi-Level, PUD', 190 : '2-Family Conversion'},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

})



#Fill in NA's



#MSZoning has 4 NA's, fill with mode

data['MSZoning'] = data['MSZoning'].fillna('RL')

#The LotFrontage Varies based on the shape of the lot, so we will fill in these NA's with the mean of each category

data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))

#Alley NA's should be 0 meaning no alley

data['Alley'] = data['Alley'].fillna(0)

#Every observation for Utilities except 1 has the same value so let's drop this column

data = data.drop('Utilities', 1)

#Exterior1st and 2nd missing 1 observation, fill in with mode

data['Exterior1st'] = data['Exterior1st'].fillna('VinylSd')

data['Exterior2nd'] = data['Exterior2nd'].fillna('VinylSd')

#Venear Type missing values should be None

data['MasVnrType'] = data['MasVnrType'].fillna('None')

#Venear Area Missing values should be 0

data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

#Missing Data for Basement columns means no basement, fill in with 0's

data['BsmtQual'] = data['BsmtQual'].fillna(0)

data['BsmtCond'] = data['BsmtCond'].fillna(0)

data['BsmtExposure'] = data['BsmtExposure'].fillna(0)

data['BsmtFinType1'] = data['BsmtFinType1'].fillna(0)

data['BsmtFinType2'] = data['BsmtFinType2'].fillna(0)

data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0)

data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0)

data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(0)

data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(0)

#One missing data for electrical, fill in with most common type

data['Electrical'] = data['Electrical'].fillna('SBrkr')

#Basement Bath's NA's put 0

data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)

data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0)

#Put in mean of KitchenQual based on neighborhood

data['KitchenQual'] = data.groupby('Neighborhood')['KitchenQual'].transform(lambda x: x.fillna(x.mean()))

#Functional, a huge majority are 'Typ' so fill with this

data['Functional'] = data['Functional'].fillna(8)

#Fireplace Quality NA's mean no Fireplace

data['FireplaceQu'] = data['FireplaceQu'].fillna(0)

#Garage NA's mean no garage

data['GarageType'] = data['GarageType'].fillna(0)

data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)

data['GarageFinish'] = data['GarageFinish'].fillna(0)

data['GarageCars'] = data['GarageCars'].fillna(0)

data['GarageArea'] = data['GarageArea'].fillna(0)

data['GarageQual'] = data['GarageQual'].fillna(0)

data['GarageCond'] = data['GarageCond'].fillna(0)

#PoolQC NA's means no pool, make NA's 0

data['PoolQC'] = data['PoolQC'].fillna(0)

#Fence NA's mean no fence, make 0

data['Fence'] = data['Fence'].fillna(0)

#MiscFeature NA's should be 0

data['MiscFeature'] = data['MiscFeature'].fillna(0)

#SaleType, will fill with mode

data['SaleType'] = data['SaleType'].fillna('WD')

#Combine any columns

data['OverallScore'] = (data['OverallQual'] + data['OverallCond']) / 2

data['ExterScore'] = (data['ExterQual'] + data['ExterCond']) / 2

data['BsmtScore'] = (data['BsmtQual'] + data['BsmtCond']) / 2

data['GarageScore'] = (data['GarageQual'] + data['GarageCond']) / 2

data['HouseSqFt'] = (data['TotalBsmtSF'] + data['GrLivArea'])

data['TotalBath'] = data['BsmtFullBath'] + (data['BsmtHalfBath'] / 2) + data['FullBath'] + (data['HalfBath'] / 2)

data['GarageRating'] = data['GarageCars'] * data['GarageFinish']
#Identify numerical and categorical variables

num_vars = data.dtypes[data.dtypes != "object"].index

cat_vars = data.dtypes[data.dtypes == 'object'].index

#Split data into numerical and categorical variables

data_num = data[num_vars]

data_cat = data[cat_vars]

#Determine variables that are highly skewed, arbitrarily select .75 as threshold

skewed = data[num_vars].apply(lambda x: skew(x))

skewed = skewed[skewed > 0.75]

skewed = skewed.index

#Log transform these skewed variables

data_num[skewed] = np.log1p(data_num[skewed])

#Create dummy variables

data_cat = pd.get_dummies(data_cat)



#Concatenate the categorical and numerical variables back into one dataset

data = pd.concat([data_num, data_cat], axis = 1)



#function to calculate RMSE



def rmse_train(model):

    rmse = np.sqrt(-cross_val_score(model, train, y, scoring = 'neg_mean_squared_error', cv = 5))

    return(rmse)



def rmse_test(model):

    rmse = np.sqrt(-cross_val_score(model, X_test, y_test, scoring = 'neg_mean_squared_error', cv = 5))

    return(rmse)

#Split the data back into the training and test sets

train = data[:train.shape[0]]

test = data[train.shape[0]:]



#Standardize the data

scaler = StandardScaler()

#Fit and transform the scaler on the training set, transform the test set

train.loc[:, num_vars] = scaler.fit_transform(train.loc[:, num_vars])

test.loc[:,num_vars] = scaler.transform(test.loc[:, num_vars])

#Alpha is the parameter to tune in Ridge Regression. Choose values of alpha

model_ridge = RidgeCV(alphas = [25, 10, 5, 1, 0.1, .05, .01, 0.001]).fit(train, y)

print(rmse_train(model_ridge).mean())

#print(rmse_test(model_ridge).mean())



#See what alpha was used

ridge_alpha = model_ridge.alpha_

print(ridge_alpha)



#Try finding a better alpha

new_ridge = RidgeCV(alphas = [ridge_alpha * .6, ridge_alpha * .75, ridge_alpha * .9, 

                              ridge_alpha * 1.1, ridge_alpha * 1.25, ridge_alpha * 1.5, 

                              ridge_alpha * 1.75, ridge_alpha * 2]).fit(train, y)

print(rmse_train(new_ridge).mean())

#print(rmse_test(new_ridge).mean())

new_alpha = new_ridge.alpha_

print(new_alpha)



#Show important features

features = model_ridge.coef_

features = pd.Series(features, index = train.columns)

best_worst_feat = pd.concat([features.sort_values().head(10), features.sort_values().tail(10)])



best_worst_feat.plot(kind = 'barh')

plt.show()
#Lasso Regression

model_lasso = LassoCV(alphas = [25, 10, 5, 1, 0.1, .05, .01, 0.001]).fit(train, y)

lasso_alpha = model_lasso.alpha_

print(lasso_alpha)

new_lasso = LassoCV(alphas = [lasso_alpha * .6, lasso_alpha * .75, lasso_alpha * .9, lasso_alpha * 1.1,

                              lasso_alpha * 1.25, lasso_alpha * 1.5, lasso_alpha * 1.75]).fit(train, y)

new_alpha = new_lasso.alpha_

print(new_alpha)



print(rmse_train(model_lasso).mean())

#print(rmse_test(model_lasso).mean())

print(rmse_train(new_lasso).mean())

#print(rmse_test(new_lasso).mean())
y_pred = new_lasso.predict(test)

y_pred = np.exp(y_pred)



tests = pd.read_csv('../input/test.csv')

submission = pd.DataFrame(y_pred, index = tests['Id'], columns = ['SalePrice'])



submission.to_csv('submission.csv')

            
elastic_model = ElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1], cv = 5, normalize = False).fit(train, y)

print(rmse_train(elastic_model).mean())



elastic_pred = np.exp(elastic_model.predict(test))

#submission = pd.DataFrame(y_pred, index = tests['Id'], columns = ['SalePrice'])

#submission.to_csv('submission.csv')
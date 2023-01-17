# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix, accuracy_score

from sklearn.model_selection import KFold,cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline

from scipy.special import boxcox1p

from xgboost import XGBRegressor
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
print("Training set size:", train.shape)

print("Test Set size: ", test.shape)
print("Train set columns not  in test set")

print([train_col for train_col in train.columns if train_col not in test.columns ])

print ("Test set columns not in test set")

print ([test_col for test_col in test.columns if test_col not in train.columns])
#View first 10 rows of data in test and training

print(train.head(10))

print(test.head(10))
train.describe()
train_ID = train['Id']

test_ID = test['Id']
train.drop(['Id'], axis = 1, inplace = True)

test.drop(['Id'], axis = 1, inplace = True)
cormatrix = train.corr()

corplot = plt.subplots(figsize =(15,12))

sns.heatmap(cormatrix, vmin = -1, vmax = 1,cbar = True, square = True, cmap = 'coolwarm')
k = 10

corr_cols = cormatrix.nlargest(k, 'SalePrice')['SalePrice'].index

cormatrix2 = np.corrcoef(train[corr_cols].values.T)

sns.set(font_scale = 1)

sns.heatmap(cormatrix2,cbar = True, square = True , cmap = 'coolwarm',annot_kws={'size': 10}, annot = True , xticklabels = corr_cols.values, yticklabels = corr_cols.values)
#ScatterPlot Matrix of most correlated features

sns.set(style = 'ticks', color_codes= True)

pltmatrix = sns.pairplot(train, vars = ['SalePrice','OverallQual', 'GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt'])

#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
sns.distplot(train['SalePrice'] , fit=norm)
train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice'] , fit=norm);
train_num = train.shape[0]

test_num = test.shape[0]

y_train = train.SalePrice.values
data = pd.concat((train, test)).reset_index(drop=True)

data.drop(['SalePrice'], axis=1, inplace=True)

print("Concatenated dataframe size is :", (data.shape))
#sum missing data calculate percentage missing in each column 

missing_df = pd.DataFrame({'Total':train.isnull().sum(), 'Percentage':(train.isnull().sum())/1460*100})

missing_df = missing_df.sort_values(by = 'Total',ascending = False)

missing_df = missing_df.loc[missing_df['Total'] > 0]

print(missing_df)
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=missing_df.index, y=missing_df['Percentage'].values)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence'):

    data.drop([col], axis = 1, inplace = True)
data.drop(['GarageArea','TotRmsAbvGrd', '1stFlrSF'], axis = 1, inplace = True)
##Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median())) 
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    data[col] = data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageCars'):

    data[col] = data[col].fillna(0)    

for col in ('BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    data[col] = data[col].fillna(0)   

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    data[col] = data[col].fillna('None')
data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
data["MasVnrType"] = data["MasVnrType"].fillna("None")

data["MasVnrArea"] = data["MasVnrArea"].fillna(0)    
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])

data.drop(['Utilities'], axis=1, inplace = True)
data["Functional"] = data["Functional"].fillna("Typ")
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0], inplace = True)



data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data.select_dtypes(include=['int64','float64']).columns
data.select_dtypes(include=['object']).columns
#MSSubClass 

data['MSSubClass'] = data['MSSubClass'].apply(str)
data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)
#Counting unique values in each of the categorical features with ordered categorical values

for col in ('ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond'):

    print(col)

    print(data[col].value_counts())



map1 = {'Po': 1, 'Fa': 2, 'TA': 3,'Gd': 4, 'Ex': 5}



for col in ('ExterCond', 'HeatingQC'):

    data[col] = data[col].map(map1)
map2 = {'None': 1, 'Po': 2, 'Fa': 3, 'TA': 4,'Gd': 5, 'Ex': 6}



for col in ('GarageCond', 'GarageQual','FireplaceQu'):

    data[col] = data[col].map(map2)
map3 = {'None':1, 'Unf': 2, 'LwQ': 3, 'Rec': 4, 'BLQ': 5,'ALQ': 6, 'GLQ': 7}



for col in ('BsmtFinType2', 'BsmtFinType1'):

    data[col] = data[col].map(map3)
map4 = { 'Fa': 1, 'TA': 2,'Gd': 3, 'Ex': 4}



for col in ('ExterQual', 'KitchenQual'):

    data[col] = data[col].map(map4)
map5 = {'None': 1, 'Fa': 2, 'TA': 3,'Gd': 4, 'Ex': 5}

data['BsmtQual'] = data['BsmtQual'].map(map5)  
map6 = {'None': 1,'Po': 2, 'Fa': 3, 'TA': 4,'Gd': 5,}

data['BsmtCond'] = data['BsmtCond'].map(map6) 
map7 = {'None': 1,'No': 2, 'Mn': 3, 'Av': 4,'Gd': 5,}

data['BsmtExposure'] = data['BsmtExposure'].map(map7) 
cat_cols = ['BldgType', 'CentralAir', 'Condition1', 'Condition2', 'Electrical',

            'Exterior1st','Exterior2nd','Foundation', 'Functional','GarageFinish','GarageType','Heating', 'HouseStyle',

            'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MSZoning','MasVnrType', 'Neighborhood', 'PavedDrive', 

            'RoofMatl', 'RoofStyle','SaleCondition', 'SaleType', 'Street','MoSold', 'YrSold',

             'MSSubClass']
data = pd.get_dummies(data, columns=cat_cols,prefix=cat_cols,drop_first=True )
num_feature = ['2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF',

               'EnclosedPorch','GrLivArea', 'GarageYrBlt','LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea',

               'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF','YearBuilt', 'YearRemodAdd']
# Check the skew of all numerical features

skewness = data[num_feature].skew()

skewness = skewness[abs(skewness) > 0.75]

skewed_feature = skewness.index
for feature in skewed_feature:

    data[feature] = np.log1p(data[feature])
X_train = data[:1458]

X_test = data[1458:]
def print_parameters(select_param, select_param_name, parameters):

    grid_search = GridSearchCV(estimator = xgb_model,

                            param_grid = parameters,

                            scoring = 'neg_mean_squared_error',

                            cv = 5,

                            n_jobs = -1)



    grid_result = grid_search.fit(X_train, y_train)



    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']

    stds = grid_result.cv_results_['std_test_score']

    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):

        print("%f (%f) with: %r" % (mean, stdev, param))
xgb_model = XGBRegressor()
learning_rate = np.arange(0.01, 0.5, 0.02)

parameters = dict(learning_rate=learning_rate)



print_parameters(learning_rate, 'learning_rate', parameters)
n_estimators = range(100, 1000, 100)

parameters = dict(n_estimators=n_estimators)



print_parameters(n_estimators, 'n_estimators', parameters)
max_depth = range(0, 5)

parameters = dict(max_depth=max_depth)



print_parameters(max_depth, 'max_depth', parameters)
subsample = np.arange(0.2, 1., 0.2)

parameters = dict(subsample=subsample)



print_parameters(subsample, 'subsample', parameters)
colsample_bytree = np.arange(0.2, 1.2, 0.2)

parameters = dict(colsample_bytree=colsample_bytree)



print_parameters(colsample_bytree, 'colsample_bytree', parameters)
gamma = np.arange(0.001, 0.1, 0.02)

parameters = dict(gamma=gamma)



print_parameters(gamma, 'gamma', parameters)
min_child_weight = np.arange(0.5, 2.0, 0.2)

parameters = dict(min_child_weight=min_child_weight)



print_parameters(min_child_weight, 'min_child_weight', parameters)
parameters = {  

                'colsample_bytree':[1],

                'subsample':[0.4,0.6],

                'gamma':[0.041],

                'min_child_weight':[1.1,1.3],

                'max_depth':[3,5],

                'learning_rate':[0.2, 0.25],

                'n_estimators':[400],                                                                    

                'reg_alpha':[0.75],

                'reg_lambda':[0.45],

                'seed':[10]

}


grid_search = GridSearchCV(estimator = xgb_model,

                        param_grid = parameters,

                        scoring = 'neg_mean_squared_error',

                        cv = 5,

                        n_jobs = -1)
xgb_model = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_

best_parameters = grid_search.best_params_
accuracies = cross_val_score(estimator=xgb_model, X=X_train, y=y_train, cv=10)
accuracies.mean()
y_pred = xgb_model.predict(X_test)

y_pred = np.floor(np.expm1(y_pred))
submission = pd.concat([test_ID, pd.Series(y_pred)], 

                        axis=1,

                        keys=['Id','SalePrice'])
submission.to_csv('sample_submission.csv', index = False)
submission
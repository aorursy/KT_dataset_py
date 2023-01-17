import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



import warnings

warnings.simplefilter(action='ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
a = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

b = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



print('The shape of the training set:', a.shape[0], 'houses and', a.shape[1], 'features.')

print('The shape of the testing set:', b.shape[0], 'houses and', b.shape[1], 'features.')

print('The testing set has 1 feature less than the training set which is the target (SalePrice) to predict.')
a.head()
num = a.select_dtypes(exclude='object')

numcorr = num[num.columns.difference(['Id'])].corr()

f, ax = plt.subplots(figsize=(17,1))

sns.heatmap(numcorr.sort_values(by=['SalePrice'], ascending=False).head(1), cmap='Blues')

plt.title("Numerical feature correlations with the SalePrice", weight='bold', fontsize=18)

plt.show()
def stylerfunc(df): 

    return df.style.background_gradient(cmap=sns.light_palette("cyan", as_cmap=True))
stylerfunc(numcorr['SalePrice'].sort_values(ascending=False).head(10).to_frame().T)
def mvrateplot(df, title):

    df = df.drop(df[df == 0].index).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))

    df.plot.barh(color='purple')

    plt.title(title, fontsize=20, weight='bold' )

    plt.show()
mvrateplot(a.isnull().mean(),'Missing value average per feature: Train set')
mvrateplot(b.isnull().mean(),'Missing value average per feature: Test set')
trainsize = a.shape[0] # the length of the train raw data set

testsize = b.shape[0]  # the length of the test raw data set

y_train = a['SalePrice'].to_frame()

#Combine train and test sets

c = pd.concat((a, b), sort=False).reset_index(drop=True)

c.name = 'c'

#Drop the target "SalePrice" and Id columns

c.drop(['SalePrice'], axis=1, inplace=True)

c.drop(['Id'], axis=1, inplace=True)

print(f"Total size of the combined dataframe {c.name} is:", c.shape)

oldlen = c.shape[1]

oldset = set(c)
c = c.dropna(thresh = len(c)*0.9, axis=1)

c.name = 'c'

print(oldlen-c.shape[1], 'features with each more than 90% of missing values are dropped from the combined dataset.')

print('The dropped features are:', list(oldset-set(c)))

print(f"--> Total size of the combined dataset {c.name} after dropping features with more than 90% M.V.:",c.shape)
mvrateplot((c.isnull().sum()/len(c)), 'Missing value average per feature')
allna = (c.isnull().sum()/len(c))

allna = allna.drop(allna[allna == 0].index).sort_values(ascending=False)



NA = c[allna.index]

NAcat = NA.select_dtypes(include='object')

NAnum = NA.select_dtypes(exclude='object')

print(f'There are {NAcat.shape[1]} categorical features with missing values')

print(f'There are {NAnum.shape[1]} numerical features with missing values')
NAnum.head()
c['MasVnrArea'] = c.MasVnrArea.fillna(0)



c['GarageYrBlt'] = c["GarageYrBlt"].fillna(c["GarageYrBlt"].median())
NAcat.head()
stylerfunc(NAcat.isnull().sum().head(18).to_frame().sort_values(by=[0]).T)
mvthreshold = 4



print(f'The categorical features having a number of M.V. less or equal to {mvthreshold}:\n', 

      list(NAcat.columns[NAcat.isnull().sum()<=mvthreshold]))
fewMVNAcat = list(NAcat.columns[NAcat.isnull().sum()<=mvthreshold])

for f in fewMVNAcat:

    c[f] = c[f].fillna(method='ffill')
stylerfunc(c[fewMVNAcat].isnull().sum().to_frame().sort_values(by=[0]).T)
mvrateplot((NAcat.isnull().sum()/len(NAcat)), 'Missing value average per categorical feature')
mvrateplot((NAnum.isnull().sum()/len(NAnum)), 'Missing value average per numerical feature')
for col in NA.columns:

    if c[col].dtype == "object":

        c[col] = c[col].fillna("None")

    else:

        c[col] = c[col].fillna(0)
c.isnull().sum().sort_values(ascending=False).head()
c['TotalArea'] = c['TotalBsmtSF'] + c['1stFlrSF'] + c['2ndFlrSF'] + c['GrLivArea'] + c['GarageArea']



c['Bathrooms'] = c['FullBath'] + c['HalfBath']*0.5 



c['YearAverage'] = (c['YearRemodAdd'] + c['YearBuilt'])/2
cb = pd.get_dummies(c) 

print("The shape of the original dataset:", c.shape)

print("The shape of the encoded dataset:", cb.shape)

print(f'--> {cb.shape[1] - c.shape[1]} encoded features are added to the combined dataset.')
Train = cb[:trainsize]  #trainsize is the number of rows of the original training set

Test = cb[trainsize:] 

print("The shape of train dataset:", Train.shape)

print("The shape of test dataset:", Test.shape)
fig = plt.figure(figsize=(15,10))

ax1 = plt.subplot2grid((2,2),(0,0))

plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('yellowgreen'))

plt.axvline(x=4600, color='r', linestyle='-')

plt.title('Ground living Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((2,2),(0,1))

plt.scatter(x=a['TotalBsmtSF'], y=a['SalePrice'], color=('red'))

plt.axvline(x=5900, color='r', linestyle='-')

plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((2,2),(1,0))

plt.scatter(x=a['1stFlrSF'], y=a['SalePrice'], color=('deepskyblue'))

plt.axvline(x=4000, color='r', linestyle='-')

plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )



ax1 = plt.subplot2grid((2,2),(1,1))

plt.scatter(x=a['MasVnrArea'], y=a['SalePrice'], color=('gold'))

plt.axvline(x=1500, color='r', linestyle='-')

plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )

plt.show()
print(a['GrLivArea'].sort_values(ascending=False).head(2))

print('*'*30)

print(a['TotalBsmtSF'].sort_values(ascending=False).head(1))

print('*'*30)

print(a['MasVnrArea'].sort_values(ascending=False).head(1))

print('*'*30)

print(a['1stFlrSF'].sort_values(ascending=False).head(1))
train = Train[(Train['GrLivArea'] < 4600) & (Train['MasVnrArea'] < 1500)]



print(f'--> {Train.shape[0]-train.shape[0]} outliers are removed from the train dataset.')
target = a[['SalePrice']]

outliers = [1298, 523, 297]

target.drop(target.index[outliers], inplace=True)
print('Make sure that both train and target sets have the same row number after removing the outliers:')

print('Train:', train.shape[0], 'rows')

print('Target:', target.shape[0], 'rows')
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((1,2),(0,0))

plt.scatter(x=a['GrLivArea'], y=a['SalePrice'], color=('yellowgreen'), alpha=0.7)

plt.title('Area-Price plot with outliers',weight='bold', fontsize=18)

plt.axvline(x=4600, color='r', linestyle='-')

#first row, second col

ax1 = plt.subplot2grid((1,2),(0,1))

plt.scatter(x=train['GrLivArea'], y=target['SalePrice'], color='navy', alpha=0.7)

plt.axvline(x=4600, color='r', linestyle='-')

plt.title('Area-Price plot without outliers',weight='bold', fontsize=18)

plt.show()
target["SalePrice"] = np.log1p(target["SalePrice"])
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(8,12))

#1 rows 2 cols

#first row, first col

ax1 = plt.subplot2grid((2,1),(0,0))

sns.distplot(np.expm1(target["SalePrice"]), color='plum')

plt.title('Before: Distribution of SalePrice',weight='bold', fontsize=18)

#first row, second col

ax1 = plt.subplot2grid((2,1),(1,0))

sns.distplot(target["SalePrice"], color='tan')

plt.title('After: Distribution of SalePrice',weight='bold', fontsize=18)

plt.show()
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

from sklearn.svm import SVR

from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from mlxtend.regressor import StackingRegressor

import math

import time
print(train.shape)

print(target.shape)

print(Test.shape)
x = train

y = np.array(target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .33, random_state=0)
scaler = RobustScaler()

# transform "x_train"

x_train = scaler.fit_transform(x_train)

# transform "x_test"

x_test = scaler.transform(x_test)

# transform "x"

x = scaler.transform(x)

# transform the test set

X_test = scaler.transform(Test)
def GridSearchCVmodels(modellist, x_train, y_train, x_test, y_test, x, y, cv): 

    results = {}

    bestregressors = {}

    for regmodel in modellist:

        vars()["grid_%s"%regmodel.__name__] = GridSearchCV(estimator=regmodel(), 

                                                           param_grid=param_grid.get(regmodel.__name__),

                                                           cv=cv, scoring='neg_mean_squared_error', verbose=0, 

                                                           n_jobs=-1, return_train_score=True, iid=False).fit(x_train, y_train.ravel())

        print(f'Regressor {modellist.index(regmodel)+1}/{len(modellist)}:')

        print("-> Best Mean RMSE: %f using %s with %s" % (-vars()["grid_%s"%regmodel.__name__].best_score_, 

                                                          regmodel.__name__, 

                                                          vars()["grid_%s"%regmodel.__name__].best_params_))



        results[regmodel.__name__] = [-max(vars()["grid_%s"%regmodel.__name__].cv_results_['mean_train_score']),

                                      -max(vars()["grid_%s"%regmodel.__name__].cv_results_['mean_test_score']),

                                      np.sqrt(mean_squared_error(vars()["grid_%s"%regmodel.__name__].best_estimator_.predict(x_train), y_train)),

                                      np.sqrt(mean_squared_error(vars()["grid_%s"%regmodel.__name__].best_estimator_.predict(x_test), y_test)), 

                                      np.sqrt(mean_squared_error(vars()["grid_%s"%regmodel.__name__].best_estimator_.predict(x), y)), 

                                     ]

        

        bestregressors[regmodel.__name__] = vars()["grid_%s"%regmodel.__name__].best_estimator_

    

    return results, bestregressors
param_grid_Ridge = {'alpha': np.arange(10.4, 10.6, 0.01), 

                    'fit_intercept': [True, False],

                   }



param_grid_Lasso = {'alpha': np.arange(4.2, 4.4, 0.01)*1e-4,

                    'fit_intercept': [True, False],

                   }



param_grid_ElasticNet = {'random_state': [1],

                         'alpha': np.arange(34, 35, 0.1)*1e-04,

                         'l1_ratio': np.arange(84, 85, 0.1)*1e-03,

                         'fit_intercept': [True, False], 

                        }



param_grid_XGBRegressor = {'objective': ['reg:squarederror'],

                           'gamma': [0], #np.arange(0, 10, 1), 

                           'n_estimators': [4110], #np.arange(4000, 4500, 100),  

                           'learning_rate': [0.01],

                           'seed': [27],

                          }



param_grid = {"Ridge": param_grid_Ridge,

              "Lasso": param_grid_Lasso,

              "ElasticNet": param_grid_ElasticNet,

              "XGBRegressor": param_grid_XGBRegressor,

              }



modellist = [Ridge, Lasso, ElasticNet, XGBRegressor]



results, bestregressors = GridSearchCVmodels(modellist, x_train, y_train, x_test, y_test, x, y, cv=5)



pd.DataFrame(results, index=['SubSubTrainData', 'SubSubTestData', 'SubTrainData', 'SubTestData', 'AllTrainData']).transpose()
def VotingRegressorCV(estimators_list, vote_params, x_train, y_train, x_test, y_test, x, y, cv):

    estimator=VotingRegressor(estimators_list)

    grid = GridSearchCV(estimator,

                        param_grid=vote_params, 

                        cv=cv, scoring='neg_mean_squared_error', 

                        verbose=0, n_jobs=-1, return_train_score=True, iid=False).fit(x_train, y_train.ravel())



    print("-> Best Mean RMSE: %f using %s with %s" % (-grid.best_score_, 

                                                      type(estimator).__name__, 

                                                      grid.best_params_))

    

    results['%s'%type(estimator).__name__] = [-max(grid.cv_results_['mean_train_score']), 

                                              -max(grid.cv_results_['mean_test_score']), 

                                              np.sqrt(mean_squared_error(grid.best_estimator_.predict(x_train), y_train)), 

                                              np.sqrt(mean_squared_error(grid.best_estimator_.predict(x_test), y_test)), 

                                              np.sqrt(mean_squared_error(grid.best_estimator_.predict(x), y)), 

                                             ]

    return grid, results
estimators_list = [('Ridge', bestregressors['Ridge']),  

                   ('Lasso', bestregressors['Lasso']), 

                   ('ElasticNet', bestregressors['ElasticNet']), 

                   ('XGBRegressor', bestregressors['XGBRegressor']), 

                  ]



#Set parameters for GridSearch

vote_params = {'weights': [(1, 1, 1, 1),

                           #(0, 1, 1, 1), 

                           #(1, 0, 1, 1), 

                           #(1, 1, 0, 1),

                           #(1, 1, 1, 0),

                          ], 

              }



grid_VotingRegressor, results = VotingRegressorCV(estimators_list, vote_params, x_train, y_train, x_test, y_test, x, y, cv=5)



pd.DataFrame(results, index=['SubSubTrainData', 'SubSubTestData', 'SubTrainData', 'SubTestData', 'AllTrainData']).transpose()
def StackingRegressorCV(regressors, meta_regressor, params, x_train, y_train, x_test, y_test, x, y, cv):

    stackreg = StackingRegressor(regressors=regressors, meta_regressor=meta_regressor, use_features_in_secondary=True)

    

    grid = GridSearchCV(estimator=stackreg,

                        param_grid=params, 

                        cv=cv, scoring='neg_mean_squared_error', 

                        verbose=0, n_jobs=-1, return_train_score=True, iid=False).fit(x_train, y_train.ravel())



    print("-> Best Mean RMSE: %f using Stack_%s with %s" % (-grid.best_score_, 

                                                         type(meta_regressor).__name__, 

                                                         grid.best_params_))

    

    results['Stack_%s'%type(meta_regressor).__name__] = [-max(grid.cv_results_['mean_train_score']), 

                                                         -max(grid.cv_results_['mean_test_score']), 

                                                         np.sqrt(mean_squared_error(grid.best_estimator_.predict(x_train), y_train)), 

                                                         np.sqrt(mean_squared_error(grid.best_estimator_.predict(x_test), y_test)), 

                                                         np.sqrt(mean_squared_error(grid.best_estimator_.predict(x), y)), 

                                                        ]

    return grid, results
params = {'meta_regressor__alpha': [63.8], #np.arange(63.8, 63.9, 0.01)*1e-0, 

          'meta_regressor__fit_intercept': [False], #[True, False], 

         } 

          

grid_Stack_Ridge, results = StackingRegressorCV(regressors = [bestregressors['Lasso'], 

                                                              bestregressors['ElasticNet'], 

                                                              grid_VotingRegressor.best_estimator_, 

                                                             ], 

                                                meta_regressor = Ridge(), 

                                                params = params, x_train = x_train, 

                                                y_train = y_train, x_test = x_test, 

                                                y_test = y_test, x = x, y = y, cv=5) 

          

pd.DataFrame(results, index=['SubSubTrainData', 'SubSubTestData', 'SubTrainData', 'SubTestData', 'AllTrainData']).transpose()
params = {'meta_regressor__random_state': [1],

          'meta_regressor__alpha': [0.0034], #np.arange(34, 35, 0.1)*1e-04,

          'meta_regressor__l1_ratio': [0.084], #np.arange(84, 85, 0.1)*1e-03,

          'meta_regressor__fit_intercept': [True, False],

         } 



grid_Stack_ElasticNet, results = StackingRegressorCV(regressors = [bestregressors['Ridge'], 

                                                                   bestregressors['Lasso'],  

                                                                   grid_VotingRegressor.best_estimator_, 

                                                                  ], 

                                                     meta_regressor = ElasticNet(), 

                                                     params = params, x_train = x_train, 

                                                     y_train = y_train, x_test = x_test, 

                                                     y_test = y_test, x = x, y = y, cv=5) 

          

pd.DataFrame(results, index=['SubSubTrainData', 'SubSubTestData', 'SubTrainData', 'SubTestData', 'AllTrainData']).transpose()
params = {'meta_regressor__seed': [27],

          'meta_regressor__objective': ['reg:squarederror'],

          'meta_regressor__gamma': [0], #np.arange(8.7, 9.2, 0.1)*1e-04,

          'meta_regressor__n_estimators': [2500], #np.arange(72, 93, 5),

         }



grid_Stack_XGBRegressor, results = StackingRegressorCV(regressors = [bestregressors['Ridge'], 

                                                                     bestregressors['Lasso'], 

                                                                     bestregressors['ElasticNet'], 

                                                                     grid_VotingRegressor.best_estimator_, 

                                                                    ], 

                                                       meta_regressor = XGBRegressor(), 

                                                       params = params, x_train = x_train, 

                                                       y_train = y_train, x_test = x_test, 

                                                       y_test = y_test, x = x, y = y, cv=5)



pd.DataFrame(results, index=['SubSubTrainData', 'SubSubTestData', 'SubTrainData', 'SubTestData', 'AllTrainData']).transpose()
df_results = pd.DataFrame(results, index=['SubSubTrainData', 'SubSubTestData', 'SubTrainData', 

                                          'SubTestData', 'AllTrainData']).transpose()

df_results.plot.barh(rot=0)

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

plt.title("RMSE score average over datasets", weight='bold', fontsize=15)

plt.show()
def blend(df):

    blenddf = pd.DataFrame({#'a': bestregressors['Lasso'].predict(df),

                            'b': grid_VotingRegressor.best_estimator_.predict(df),

                            'c': grid_Stack_XGBRegressor.best_estimator_.predict(df), 

                           })

    return blenddf



linregtest = LinearRegression(fit_intercept='False', normalize='False').fit(blend(x_test), y_test)

print(linregtest.coef_)

print(np.sqrt(mean_squared_error(linregtest.predict(blend(x_test)), y_test)))
#Submission of the results predicted by the average of Lasso/Voting/Stacking

dft = blend(X_test)



finalb1 = (np.expm1(dft)*(linregtest.coef_/linregtest.coef_.sum())).sum(axis=1)

finalb2 = np.expm1(linregtest.predict(blend(X_test)).ravel())



finalb = linregtest.predict(np.expm1(blend(X_test))).ravel()



submission = pd.DataFrame({"Id": b["Id"], 

                           "SalePrice": finalb, 

                          })

submission.to_csv("submission.csv", index=False)

submission.head()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from scipy import stats

from scipy.stats import norm, skew

import os

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge 

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print(train.head())

print('**'* 50)

print(test.head())
print(train.info())

print('**'* 50)

print(test.info())
df_num = train.select_dtypes(include = ['float64', 'int64'])



for i in range(0, len(df_num.columns), 6):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+6],

                y_vars=['SalePrice'], kind='reg')
#dropping outlayers

train.drop(train[train.GrLivArea>4000].index, inplace = True)

train.reset_index(drop = True, inplace = True)



train.drop(train[train.BsmtFinSF1>4000].index, inplace = True)

train.reset_index(drop = True, inplace = True)

      

train.drop(train[train.TotalBsmtSF>4000].index, inplace = True)

train.reset_index(drop = True, inplace = True)

         

train.drop(train[train.LotArea>200000].index, inplace = True)

train.reset_index(drop = True, inplace = True)
df_num = train.select_dtypes(include = ['float64', 'int64'])



for i in range(0, len(df_num.columns), 6):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+6],

                y_vars=['SalePrice'], kind='reg')
#heatmap for all features

plt.figure(figsize=(30,8))

sns.heatmap(train.corr(),cmap='coolwarm',annot = True)

plt.show()
#missing data for train

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
#missing data for test

total_test = test.isnull().sum().sort_values(ascending=False)

percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
# keeping only those object features which are missing not more then 80% of it's data.

# defining a function for droping invalid objects

def drop_invalid_objects(x):

    count_null = x.select_dtypes(include= 'object').isna().sum()

    count_null = count_null[count_null > 0]



    ratio = x['Id'].count()*0.2



    drop_NaN_object = pd.DataFrame(count_null[count_null > ratio])

    drop_NaN_object = drop_NaN_object.reset_index().filter(['index']).set_index('index').transpose()

    drop_NaN_object = list(drop_NaN_object)



    x.drop(columns = x[drop_NaN_object], inplace = True)



    count_null = x.select_dtypes(include= 'object').isna().sum()

    count_null = count_null[count_null > 0]

    print(count_null)



drop_invalid_objects(train)

print('='*30)

drop_invalid_objects(test)
count_null = train.select_dtypes(include= 'object').isna().sum()

count_null = pd.DataFrame(count_null[count_null > 0])

count_null = count_null.reset_index().filter(['index']).set_index('index').transpose()

count_null = list(count_null)



for column in count_null:

    train[column] = train[column].fillna(train[column].mode()[0])



for column in count_null:

    test[column] = test[column].fillna(test[column].mode()[0])    
# replacing nulls with mean of others

train.fillna(train.mean(), inplace=True)

test.fillna(test.mean(), inplace=True)



print(test.shape, train.shape)

train.info(), test.info()

train.tail()
train.isnull().sum().sort_values(ascending=False).head(20)
train.describe()
test.describe()
# Categorical boolean mask

categorical_feature_mask = train.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = train.columns[categorical_feature_mask].tolist()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

train[categorical_cols] = train[categorical_cols].apply(lambda col: labelencoder.fit_transform(col.astype(str)))
# Categorical boolean mask

categorical_feature_mask_test = test.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols_test = test.columns[categorical_feature_mask_test].tolist()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

test[categorical_cols_test] = test[categorical_cols_test].apply(lambda col: labelencoder.fit_transform(col.astype(str)))
train.head()
test.head()
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
#saleprice correlation matrix

k = 15 #number of variables for heatmap

plt.figure(figsize=(16,8))

corrmat = train.corr()

# picking the top 15 correlated features

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
test=test[cols.drop('SalePrice')]
test.head()
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mean())

test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mean())

test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mean())

test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mean())

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test.isnull().sum().sort_values(ascending=False).head(20)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice', axis=1), train['SalePrice'], test_size=0.3, random_state=101)
# we are going to scale to data



y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)

y_train = sc_X.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
X_train
X_test
linreg = LinearRegression()

parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}

grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")

grid_linreg.fit(X_train, y_train)



print("Best LinReg Model: " + str(grid_linreg.best_estimator_))

print("Best Score: " + str(grid_linreg.best_score_))
linreg = grid_linreg.best_estimator_

linreg.fit(X_train, y_train)

lin_pred = linreg.predict(X_test)

r2_lin = r2_score(y_test, lin_pred)

rmse_lin = np.sqrt(mean_squared_error(y_test, lin_pred))

print("R^2 Score: " + str(r2_lin))

print("RMSE Score: " + str(rmse_lin))

scores_lin = cross_val_score(linreg, X_train, y_train, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lin)))
lasso = Lasso()

parameters_lasso = {"fit_intercept" : [True, False], "normalize" : [True, False], "precompute" : [True, False], "copy_X" : [True, False]}

grid_lasso = GridSearchCV(lasso, parameters_lasso, verbose=1, scoring="r2")

grid_lasso.fit(X_train, y_train)



print("Best Lasso Model: " + str(grid_lasso.best_estimator_))

print("Best Score: " + str(grid_lasso.best_score_))
lasso = grid_lasso.best_estimator_

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

r2_lasso = r2_score(y_test, lasso_pred)

rmse_lasso = np.sqrt(mean_squared_error(y_test, lasso_pred))

print("R^2 Score: " + str(r2_lasso))

print("RMSE Score: " + str(rmse_lasso))
scores_lasso = cross_val_score(lasso, X_train, y_train, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_lasso)))
ridge = Ridge()

parameters_ridge = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False], "solver" : ["auto"]}

grid_ridge = GridSearchCV(ridge, parameters_ridge, verbose=1, scoring="r2")

grid_ridge.fit(X_train, y_train)



print("Best Ridge Model: " + str(grid_ridge.best_estimator_))

print("Best Score: " + str(grid_ridge.best_score_))

ridge = grid_ridge.best_estimator_

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

r2_ridge = r2_score(y_test, ridge_pred)

rmse_ridge = np.sqrt(mean_squared_error(y_test, ridge_pred))

print("R^2 Score: " + str(r2_ridge))

print("RMSE Score: " + str(rmse_ridge))
scores_ridge = cross_val_score(ridge, X_train, y_train, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_ridge)))
dtr = DecisionTreeRegressor()

parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], "splitter" : ["best", "random"], "min_samples_split" : [2, 3, 5, 10], 

                  "max_features" : ["auto", "log2"]}

grid_dtr = GridSearchCV(dtr, parameters_dtr, verbose=1, scoring="r2")

grid_dtr.fit(X_train, y_train)



print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))

print("Best Score: " + str(grid_dtr.best_score_))
dtr = grid_dtr.best_estimator_

dtr.fit(X_train, y_train)

dtr_pred = dtr.predict(X_test)

r2_dtr = r2_score(y_test, dtr_pred)

rmse_dtr = np.sqrt(mean_squared_error(y_test, dtr_pred))

print("R^2 Score: " + str(r2_dtr))

print("RMSE Score: " + str(rmse_dtr))
scores_dtr = cross_val_score(dtr, X_train, y_train, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_dtr)))
rf = RandomForestRegressor()

paremeters_rf = {"n_estimators" : [5, 10, 15, 20], "criterion" : ["mse" , "mae"], "min_samples_split" : [2, 3, 5, 10], 

                 "max_features" : ["auto", "log2"]}

grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2")

grid_rf.fit(X_train, y_train)



print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))

print("Best Score: " + str(grid_rf.best_score_))
rf = grid_rf.best_estimator_

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

r2_rf = r2_score(y_test, rf_pred)

rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))

print("R^2 Score: " + str(r2_rf))

print("RMSE Score: " + str(rmse_rf))
scores_rf = cross_val_score(rf, X_train, y_train, cv=10, scoring="r2")

print("Cross Validation Score: " + str(np.mean(scores_rf)))
model_performances = pd.DataFrame({

    "Model" : ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor"],

    "Best Score" : [grid_linreg.best_score_,  grid_ridge.best_score_, grid_lasso.best_score_, grid_dtr.best_score_, grid_rf.best_score_],

    "R Squared" : [str(r2_lin)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5], str(r2_dtr)[0:5], str(r2_rf)[0:5]],

    "RMSE" : [str(rmse_lin)[0:8], str(rmse_ridge)[0:8], str(rmse_lasso)[0:8], str(rmse_dtr)[0:8], str(rmse_rf)[0:8]]

})

model_performances.round(4)



print("Sorted by Best Score:")

model_performances.sort_values(by="Best Score", ascending=False)
print("Sorted by RMSE:")

model_performances.sort_values(by="RMSE", ascending=True)
ridge.fit(X_train, y_train)
X_test.reshape(-1)
submission = ridge.predict(X_test)
ridge.predict(X_test)
submission = pd.DataFrame(submission, columns=['SalePrice']).to_csv('submission.csv', index=False)
submission = pd.DataFrame({

        "Id": ["Id"],

        "SalePrice": submission

    })



submission.to_csv("prices.csv", index=False)
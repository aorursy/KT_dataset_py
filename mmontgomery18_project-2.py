import numpy as np 

import pandas as pd 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(20)
all_data = pd.concat((train.loc[:,"MSSubClass":"SaleCondition"],

                      test.loc[:,"MSSubClass":"SaleCondition"]))
cm = train.corr()["SalePrice"].sort_values(ascending=False)

cm.head(20)
import seaborn as sns

import matplotlib.pyplot as plt

cm=train[["SalePrice","OverallQual","GrLivArea","GarageCars",

                  "GarageArea","GarageYrBlt","TotalBsmtSF","1stFlrSF","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]].corr()

plt.subplots(figsize=(6, 4))

sns.heatmap(cm, vmax=1, square=False);
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show();
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train['SalePrice'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice']);
total = train.isnull().sum().sort_values(ascending=False)

total.head(20)
train = train.drop(['MiscFeature','Alley','Fence','PoolQC'],axis = 1)

test = test.drop(['MiscFeature','Alley','Fence','PoolQC'],axis = 1)

all_data = all_data.drop(['MiscFeature','Alley','Fence','PoolQC'],axis = 1)
train = train.fillna(train.mean())

test = test.fillna(test.mean())
train['LivArea_Total'] = train['GrLivArea'] + train['GarageArea']  

test['LivArea_Total'] = test['GrLivArea'] + test['GarageArea']

train[['LivArea_Total','GrLivArea','GarageArea']].head()
train['FlrSF_Total'] = train['1stFlrSF']  + train['TotalBsmtSF']

test['FlrSF_Total'] = test['1stFlrSF'] + test['TotalBsmtSF']

train[['FlrSF_Total','1stFlrSF','2ndFlrSF']].head()
train = pd.get_dummies(train)

test = pd.get_dummies(test)

train.head()
all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())
from sklearn.linear_model import Lasso

lasso = Lasso()





X = train.drop(['SalePrice'], axis = 1)

y = train['SalePrice']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20,random_state = 101)



lasso = lasso.fit(X_train, y_train)



y_pred_lasso = lasso.predict(X_test)





from sklearn import metrics



print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred_lasso)))

print('R^2=',metrics.explained_variance_score(y_test,y_pred_lasso))

print('Accuracy Train', lasso.score(X_train, y_train ))

print('Accuracy Test', lasso.score(X_test, y_test))



pd.set_option('display.float_format', lambda x: '%.2f' % x)

cdf = pd.DataFrame(data = lasso.coef_,index = X_train.columns, columns = ['Lasso Coefficients'])

# **RANDOM FOREST**

cdf.sort_values(by = 'Lasso Coefficients', ascending = False)

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.linear_model import LassoCV

from sklearn.model_selection import cross_val_score



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 7))

    return(rmse)
model_lasso = LassoCV(alphas = [10, 1, 0.1, 0.001, .0005]).fit(X_train, y)

rmse_cv(model_lasso).mean()

from sklearn.ensemble import RandomForestRegressor



X_best = train[["OverallQual","LivArea_Total",

                  "GarageArea","GarageYrBlt","FlrSF_Total","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]

#y = train['SalePrice']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_best,y, test_size = .50,random_state = 101)



rforest = RandomForestRegressor(n_estimators = 300, random_state = 0) 

rforest.fit(X_best,y)



y_pred_rforest = rforest.predict(X_test)



from sklearn import metrics

print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, y_pred_rforest)))

print('R^2 =',metrics.explained_variance_score(y_test,y_pred_rforest))

print('Accuracy Train', rforest.score(X_train, y_train ))

print('Accuracy Test', rforest.score(X_test, y_test))
X_best = X_train[["OverallQual","GrLivArea",

                  "GarageArea","GarageYrBlt","1stFlrSF","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]



rforest = RandomForestRegressor(n_estimators = 300, random_state = 0) 

rforest.fit(X_best,y)



y_pred_rforest = rforest.predict(X_best)



print(rmse_cv(rforest).mean())

train = train.fillna(train.mean())

test = test.fillna(test.mean())



X_train2 = train[["OverallQual","LivArea_Total",

                  "GarageArea","GarageYrBlt","FlrSF_Total","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]





y_train2 = train['SalePrice']





X_test2 = test[["OverallQual","LivArea_Total",

                  "GarageArea","GarageYrBlt","FlrSF_Total","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]









print('Data Shapes')

print('x_train shape', X_train2.shape)

print('y_train shape',y_train2.shape)

print('x_test shape', X_test2.shape)







rforest = RandomForestRegressor(n_estimators = 300, random_state = 0) 

rforest.fit(X_train2,y_train2)





y_pred_rforest2 = rforest.predict(X_test2)
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": y_pred_rforest2

    })



submission.to_csv('HousePricesRF1.csv', index=False)
from sklearn.neighbors import KNeighborsRegressor





train_X, test_X, train_y, test_y = train_test_split(X, y, 

                                                    train_size=0.5,

                                                    test_size=0.5,

                                                    random_state=123)



neigh = KNeighborsRegressor(n_neighbors = 5)

neigh.fit(train_X, train_y)



knn_pred_y = neigh.predict(test_X)



print('RMSE',np.sqrt(metrics.mean_squared_error(test_y, knn_pred_y)))

print('R^2 =',metrics.explained_variance_score(test_y,knn_pred_y))

print('Accuracy Train', neigh.score(train_X, train_y ))

print('Accuracy Test', neigh.score(test_X, test_y))



train_X2 = train[["OverallQual","LivArea_Total",

                  "GarageArea","GarageYrBlt","FlrSF_Total","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]





train_y2 = train['SalePrice']





test_X2 = test[["OverallQual","LivArea_Total",

                  "GarageArea","GarageYrBlt","FlrSF_Total","FullBath",

                  "TotRmsAbvGrd","YearBuilt","YearRemodAdd"]]









print('Data Shapes')

print('x_train shape', train_X2.shape)

print('y_train shape',train_y2.shape)

print('x_test shape', test_X2.shape)







neigh = KNeighborsRegressor(n_neighbors = 5)

neigh.fit(train_X2, train_y2)











knn_pred_y2 = neigh.predict(test_X2)
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": knn_pred_y2

    })



submission.to_csv('HousePricesKNN.csv', index=False)
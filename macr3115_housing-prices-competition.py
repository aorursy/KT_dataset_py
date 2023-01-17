# imports for processing

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats

import sklearn



# visualization imports

import matplotlib.pyplot as plt

import seaborn as sns



# model

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import cross_val_score



# list available files

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# import training data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_all = pd.concat((df_train.loc[:, 'Id':'SaleCondition'],  # all but SalePrice

                    df_test.loc[:, 'Id':'SaleCondition']))
# heatmap data to determine redundant features

correlation_matrix = df_train.corr(method='pearson')

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(correlation_matrix, vmax=0.8, square=True);
# based on heatmap, exclude TotRmsAbvGrd, GarageArea, 1stFlrSF since their twins 

# (TotalBsmtSF, GarageCars, and GrLivArea) are included

df_all.drop('TotRmsAbvGrd', 1, inplace=True)

df_all.drop('GarageArea', 1, inplace=True)

df_all.drop('1stFlrSF', 1, inplace=True)
# list features in training data with the most missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# drop top 18 columns in table (visual analysis of these features elsewhere shows a low 

# correlation with SalePrice, and much missing data)



df_all = df_all.drop((missing_data[missing_data['Total'] > 1]).index, 1)

df_all.isnull().sum().max()  # check for additional missing data
# fill missing data in df_test with most common data value; check for missing data

df_all = df_all.fillna(df_all.mode().iloc[0])

df_all.isnull().sum().max()  # make sure there's no more missing data
# check out the SalePrice histogram and normal probability plot for skew and other 

# deviations from normal

sns.distplot(df_train['SalePrice'], fit=scipy.stats.norm);

fig = plt.figure()

res = scipy.stats.probplot(df_train['SalePrice'], plot=plt)
# skew and deviations can be improved with a log transformation:

df_train['SalePrice'] = np.log(df_train['SalePrice'])
# make sure SalePrice column will still match training data columns

x_train = df_all[:df_train.shape[0]]

print(np.shape(x_train))

print(np.shape(df_train))
# check the plots again and see how SalePrice has changed

sns.distplot(df_train['SalePrice'], fit=scipy.stats.norm);

fig = plt.figure()

res = scipy.stats.probplot(df_train['SalePrice'], plot=plt)
# GrLivArea is continuous and highly correlated with SalePrice (see heatmap), so check it too

sns.distplot(df_all['GrLivArea'], fit=scipy.stats.norm);

fig = plt.figure() 

res = scipy.stats.probplot(df_all['GrLivArea'], plot=plt)
# apply same transformation as above to improve skew

df_all['GrLivArea'] = np.log(df_all['GrLivArea'])
# plot again to check

sns.distplot(df_all['GrLivArea'], fit=scipy.stats.norm);

fig = plt.figure()

res = scipy.stats.probplot(df_all['GrLivArea'], plot=plt)
# make categorical variables into dummy vars

df_all = pd.get_dummies(df_all)
# make matrices for sklearn

y = df_train.SalePrice

x_train = df_all[:df_train.shape[0]]

x_test = df_all[df_train.shape[0]:]



# check shapes

print(np.shape(y))

print(np.shape(x_train))

print(np.shape(x_test))
# root mean square error with cross validation of 5

def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, x_train, y, scoring="neg_mean_squared_error", cv=5))

    return(rmse)
# choose some alphas for testing and get errors of each

ridge_alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_rmse_kridge = [rmse_cv(KernelRidge(alpha=alpha)).mean() for alpha in ridge_alphas]
# check out in list form

print(cv_rmse_kridge)
# graph to see minimum more easily

cv_kridge = pd.Series(cv_rmse_kridge, index = ridge_alphas)

cv_kridge.plot(title = "Validation: Kernel Ridge")

plt.xlabel("Alpha Values")

plt.ylabel("RMSE")
cv_kridge.min()
# choose alpha = 5 to minimize RMSE

model_kridge = KernelRidge(alpha = 5).fit(x_train, y)
# apply model to test data

kridge_preds = np.expm1(model_kridge.predict(x_test))
# check out predictions before downloading

print(kridge_preds)
solution = pd.DataFrame({"Id":df_test.Id, "SalePrice":kridge_preds})

solution.to_csv("kernel_ridge_regression.csv", index = False)
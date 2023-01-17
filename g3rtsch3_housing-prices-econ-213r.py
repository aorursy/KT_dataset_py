import pandas as pd

import requests 

import numpy as np

import json

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn

from scipy import stats

from scipy.stats import norm, skew



print('The scikit-learn version is {}.'.format(sklearn.__version__))



import warnings 

warnings.filterwarnings('ignore')
test_data = pd.read_csv("../input/test.csv")

test_data.head()
test_data.shape
ames_df = pd.read_csv("../input/train.csv")

ames_df.head()
ames_df.shape
# check a categorical explanatory variable

var = 'OverallQual'

data = pd.concat([ames_df['SalePrice'], ames_df[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data = data)



# check a quantitative explanatory variable

var = 'GrLivArea'

data = pd.concat([ames_df['SalePrice'], ames_df[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000));



# year built will likely be an important variable, so check this one too

var = 'YearBuilt'

data = pd.concat([ames_df['SalePrice'], ames_df[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

plt.xticks(rotation=90);



# When were houses sold?

ames_df.groupby(['YrSold','MoSold']).Id.count().plot(kind='bar', figsize=(14,4))

plt.title('When were houses sold?')

plt.show()



# When were the houses built?

print('Oldest house built in {}. Newest house built in {}.'.format(

    ames_df.YearBuilt.min(), ames_df.YearBuilt.max()))

ames_df.YearBuilt.hist(bins=14, rwidth=.9, figsize=(12,4))

plt.title('When were the houses built?')

def get_feature_groups():

    # Numerical Features

    num_features = ames_df.select_dtypes(include=['int64','float64']).columns

    num_features = num_features.drop(['Id','SalePrice']) # drop ID and SalePrice



    # Categorical Features

    cat_features = ames_df.select_dtypes(include=['object']).columns

    return list(num_features), list(cat_features)



num_features, cat_features = get_feature_groups()



# Grid of distribution plots of all numerical features

f = pd.melt(ames_df, value_vars=sorted(num_features))

g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)

g = g.map(sns.distplot, 'value')



# Grid of frequency plots of all categoriccal features

f = pd.melt(ames_df, value_vars=sorted(cat_features))

g = sns.FacetGrid(f, col='variable', col_wrap=4, sharex=False, sharey=False)

plt.xticks(rotation='vertical')

g = g.map(sns.countplot, 'value')

[plt.setp(ax.get_xticklabels(), rotation=60) for ax in g.axes.flat]

g.fig.tight_layout()

plt.show()
ames_df['SalePrice'].describe()

sns.distplot(ames_df['SalePrice'])

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



# draw the qq plot to check for normality

fig = plt.figure()

res = stats.probplot(ames_df['SalePrice'], plot=plt)

plt.show()



# the qqplot does not show linearity, so we need to transform SalePrice
# log transformation

ames_df["SalePrice"] = np.log1p(ames_df["SalePrice"])



#Check the new distribution 

sns.distplot(ames_df['SalePrice'] , fit=norm);



# check the new qq plot

fig = plt.figure()

res = stats.probplot(ames_df['SalePrice'], plot=plt)

plt.show()



# much more linear post-log transformation



print("Find most important features relative to target")

corr = ames_df.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
# store the logged salesprice

yT = ames_df["SalePrice"]

print(yT[1:5])
ames_df.shape
test_data.shape
all_data = ames_df

all_data.head()

all_data.shape
corr = all_data.corr()

sns.heatmap(corr)
alltrain = all_data.drop(['SalePrice'], axis=1)

traintest = alltrain.append(test_data)



# convert using dummies

traintest = pd.get_dummies(traintest)

dataLength = all_data.shape[0]

all_data_num = traintest[0:dataLength]

test_data_num = traintest[dataLength:]



# fill missing values

all_data_num = all_data_num.fillna(method = "bfill")

test_data_num = test_data_num.fillna(method = "bfill")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(all_data_num)

all_data_num = scaler.transform(all_data_num)



scaler.fit(test_data_num)

test_data_num = scaler.transform(test_data_num)
print(test_data_num.shape, all_data_num.shape)
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



X_train = all_data_num

X_train.shape



# split the TRAINING DATA further into training and testing

# 80% of the original training remains train, and 20% becomes test

Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, yT, test_size = 0.2, random_state = 23)
print("Xtrain : " + str(Xtrain.shape))

print("Xtest : " + str(Xtest.shape))

print("ytrain : " + str(ytrain.shape))

print("ytest : " + str(ytest.shape))
from sklearn.linear_model import LinearRegression



linear_regression_model = SGDRegressor(tol=.0001, eta0=.01)

linear_regression_model.fit(Xtrain, ytrain)

predictions = linear_regression_model.predict(Xtrain)

mse = mean_squared_error(ytrain, predictions)

print("RMSE: {}".format(np.sqrt(mse)))
linear_regression_model = SGDRegressor(tol=.0001, eta0=.01)

linear_regression_model.fit(Xtrain, ytrain)

train_predictions = linear_regression_model.predict(Xtrain)

test_predictions = linear_regression_model.predict(Xtest)



print("Train MSE: {}".format(mean_squared_error(ytrain, train_predictions)))

print("Test MSE: {}".format(mean_squared_error(ytest, test_predictions)))
from sklearn.linear_model import ElasticNetCV



# the alphas tell us how much to weight regularization

clf = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=[.1, 1, 10])

clf.fit(Xtrain, ytrain)

train_predictions = clf.predict(Xtrain)

test_predictions = clf.predict(Xtest)

print("Train MSE: {}".format(mean_squared_error(ytrain, train_predictions)))

print("Test MSE: {}".format(mean_squared_error(ytest, test_predictions)))
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



# iterate to find the best value of alpha

listPara=[0.0001, 0.001, 0.01, 1, 10, 100, 200, 300, 400, 500]

# record the error for each parameter chosen

error = []

for i in range(len(listPara)):

    # apply ridge regression

    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)

    ridgeReg.fit(Xtrain, ytrain)

    optPredict = ridgeReg.predict(Xtest)

    # calculate the error

    error.append(np.sqrt(mean_squared_error(optPredict, ytest)))

# plot the error

plt.scatter(x=listPara, y=error, color='b', marker='*')

# calculate the best parameter in list

print(pd.Series(data=error, index=listPara).idxmin())
# iterate through the parameter list again

listPara = np.arange(200, 400, 10)



error = []

for i in range(len(listPara)):

    # apply ridge regression

    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)

    ridgeReg.fit(Xtrain, ytrain)

    optPredict = ridgeReg.predict(Xtest)

    # calculate the error

    error.append(np.sqrt(mean_squared_error(optPredict, ytest)))



plt.scatter(x=listPara, y=error, color='b', marker='*')

# calculate the best parameter in list

print(pd.Series(data=error, index=listPara).idxmin())
# iterate through the parameter list again

listPara = np.arange(290, 310, .1)

error = []

for i in range(len(listPara)):

    # apply ridge regression

    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)

    ridgeReg.fit(Xtrain, ytrain)

    optPredict = ridgeReg.predict(Xtest)

    # calculate the error

    error.append(np.sqrt(mean_squared_error(optPredict, ytest)))



plt.scatter(x=listPara, y=error, color='b', marker='*')

# calculate the best parameter in list

print(pd.Series(data=error, index=listPara).idxmin())
# fit the model with the chosen value of alpha = 298

ridge = Ridge(alpha=298)

# use all training data available

ridgeReg.fit(Xtrain, ytrain)

train_predictionsR = ridgeReg.predict(Xtrain)

test_predictionsR = ridgeReg.predict(Xtest)



print("Train MSE: {}".format(mean_squared_error(ytrain, train_predictionsR)))

print("Test MSE: {}".format(mean_squared_error(ytest, test_predictionsR)))
clf = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], alphas=[.1, 1, 10])

clf.fit(X_train, yT)

test_predictions = clf.predict(test_data_num)

print(test_predictions[1:5])



# back-transform the log-transformation

test_predictions = np.exp(list(test_predictions))-1

print(test_predictions[1:5])
my_submission = pd.DataFrame({'Id': test_data.iloc[:, 0], 'SalePrice': test_predictions})

my_submission.to_csv('submission.csv', index = False)
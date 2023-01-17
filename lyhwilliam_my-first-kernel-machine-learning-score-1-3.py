# import necessary packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

# use seaborn to costomize matplotlib plot theme

import seaborn as sns

sns.set_style("darkgrid")

import scipy.stats as stats

# use ridge regession library provided by sklearn

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
# filter unnecessary warnings

# thanks to Ahmad Javed for teaching me this method!

import warnings 

warnings.filterwarnings('ignore')
# load data

dataTrain=pd.read_csv('../input/train.csv')

dataTest=pd.read_csv('../input/test.csv')
print(dataTrain.head(), '\n', dataTrain.shape)

print(100 * '*')

print(dataTest.head(), '\n', dataTest.shape)
# check possible outliers in the dependent variable, SalePrice

plt.scatter(x='GrLivArea', y='SalePrice', data=dataTrain, color='b', marker='*')

plt.show()
print('before removal: ', dataTrain.shape)

# remove two outliers

dataTrain.drop(dataTrain[(dataTrain['GrLivArea']>4000) & (dataTrain['SalePrice']<200000)].index, inplace=True)

print('after removal: ', dataTrain.shape)
# check NaN values in the training and test set

print(dataTrain.isnull().sum().sort_values(ascending=False).head(25))

print(100*'*')

print(dataTest.isnull().sum().sort_values(ascending=False).head(35))
# record all train columns

dicColumn = dataTrain.columns

for nameColumn in dicColumn:

    # object features

    if dataTrain[nameColumn].dtype == 'object':

        # value that should be 'NA', meaning 'no'

        if dataTrain[nameColumn].isnull().sum()>30:

            dataTrain[nameColumn].fillna(value='NA', inplace=True)

        # value that is unavailable

        else:

            dataTrain[nameColumn].fillna(value=dataTrain[nameColumn].mode()[0], inplace=True)

    # numerical features

    else:

        if dataTrain[nameColumn].isnull().any():

            # fill with median

            dataTrain[nameColumn].fillna(value=dataTrain[nameColumn].median(), inplace=True)

            

# record all test columns

dicColumnT = dataTest.columns

for nameColumn in dicColumnT:

    # object features

    if dataTest[nameColumn].dtype == 'object':

        # value that should be 'NA', meaning 'no'

        if dataTest[nameColumn].isnull().sum()>30:

            dataTest[nameColumn].fillna(value='NA', inplace=True)

        # value that is unavailable

        else:

            dataTest[nameColumn].fillna(value=dataTest[nameColumn].mode()[0], inplace=True)

    # numerical features

    else:

        if dataTest[nameColumn].isnull().any():

            # fill with median

            dataTest[nameColumn].fillna(value=dataTest[nameColumn].median(), inplace=True)

            

# check NaN values in the training and test set

print(dataTrain.isnull().sum().sort_values(ascending=False).head(10))

print(100*'*')

print(dataTest.isnull().sum().sort_values(ascending=False).head(10))
plt.figure(figsize=(15,12))

sns.heatmap(dataTrain.corr(), vmax=0.9)
# correlation between features

print(dataTrain.corr()['YearBuilt']['GarageYrBlt'])

# correlation between features and saleprice

print(dataTrain.corr()['SalePrice']['YearBuilt'])

print(dataTrain.corr()['SalePrice']['GarageYrBlt'])
# correlation between features

print(dataTrain.corr()['GarageArea']['GarageCars'])

# correlation between features and saleprice

print(dataTrain.corr()['SalePrice']['GarageArea'])

print(dataTrain.corr()['SalePrice']['GarageCars'])
# correlation between features

print(dataTrain.corr()['TotalBsmtSF']['1stFlrSF'])

# correlation between features and saleprice

print(dataTrain.corr()['SalePrice']['TotalBsmtSF'])

print(dataTrain.corr()['SalePrice']['1stFlrSF'])
# correlation between features

print(dataTrain.corr()['GrLivArea']['TotRmsAbvGrd'])

# correlation between features and saleprice

print(dataTrain.corr()['SalePrice']['GrLivArea'])

print(dataTrain.corr()['SalePrice']['TotRmsAbvGrd'])
# correlation between features

print(dataTrain.corr()['GrLivArea']['TotRmsAbvGrd'])

# correlation between features and saleprice

print(dataTrain.corr()['SalePrice']['GrLivArea'])

print(dataTrain.corr()['SalePrice']['TotRmsAbvGrd'])
print(dataTrain.shape, dataTest.shape)

dataTrain.drop(['GarageYrBlt','GarageArea','1stFlrSF','TotRmsAbvGrd'], axis=1, inplace=True)

dataTest.drop(['GarageYrBlt','GarageArea','1stFlrSF','TotRmsAbvGrd'], axis=1, inplace=True)

print(dataTrain.shape, dataTest.shape)
# correlation between SalePrice and features in ascending order

print(dataTrain.corr()['SalePrice'].abs().sort_values(ascending=True).head(25))
print(dataTrain.shape, dataTest.shape)

# delete features w=with correlation factors less than 0.3

irrelatedCol = dataTrain.corr()['SalePrice'].abs().sort_values(ascending=True).head(19).index

dataTrain.drop(irrelatedCol, axis=1, inplace=True)

dataTest.drop(irrelatedCol, axis=1, inplace=True)

print(dataTrain.shape, dataTest.shape)
# check skewness in dependent variable

sns.distplot(dataTrain['SalePrice'])
# log transformation

dataTrain['SalePrice']=np.log1p(dataTrain['SalePrice'])

sns.distplot(dataTrain['SalePrice'])
xTrainData = dataTrain.drop(['SalePrice'], axis=1)

inputData = xTrainData.append(dataTest)

# convert using hot encoding

inputData = pd.get_dummies(inputData)

xTrainLenth = xTrainData.shape[0]

xTrainData = inputData[0:xTrainLenth]

xTestData = inputData[xTrainLenth:]

print(xTrainData.shape, xTestData.shape)

# split the training set into set for model building and set for model optimizing

xTrain, xOpt, yTrain, yOpt = train_test_split(xTrainData, dataTrain['SalePrice'], test_size=0.3, random_state=100)
# iterate throw the parameter list for the best value of parameter alpha

listPara=[0.0001, 0.001, 0.01, 1, 10, 100, 1000]

# record the error for each parameter chosen

error = []

for i in range(len(listPara)):

    # apply ridge regression

    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)

    ridgeReg.fit(xTrain, yTrain)

    optPredict = ridgeReg.predict(xOpt)

    # calculate the error

    error.append(np.sqrt(mean_squared_error(optPredict, yOpt)))

# plot the error

plt.scatter(x=listPara, y=error, color='b', marker='*')

# calculate the best parameter in list

print(pd.Series(data=error, index=listPara).idxmin())
# iterate throw the parameter list for the best value of parameter alpha

listPara = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# record the error for each parameter chosen

error = []

for i in range(len(listPara)):

    # apply ridge regression

    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)

    ridgeReg.fit(xTrain, yTrain)

    optPredict = ridgeReg.predict(xOpt)

    # calculate the error

    error.append(np.sqrt(mean_squared_error(optPredict, yOpt)))

# plot the error

plt.scatter(x=listPara, y=error, color='b', marker='*')

# calculate the best parameter in list

print(pd.Series(data=error, index=listPara).idxmin())
# iterate throw the parameter list for the best value of parameter alpha

listPara = np.arange(1, 21, 1)

# record the error for each parameter chosen

error = []

for i in range(len(listPara)):

    # apply ridge regression

    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)

    ridgeReg.fit(xTrain, yTrain)

    optPredict = ridgeReg.predict(xOpt)

    # calculate the error

    error.append(np.sqrt(mean_squared_error(optPredict, yOpt)))

# plot the error

plt.scatter(x=listPara, y=error, color='b', marker='*')

# calculate the best parameter in list

print(pd.Series(data=error, index=listPara).idxmin())
# iterate throw the parameter list for the best value of parameter alpha

listPara = np.arange(6, 8, 0.1)

# record the error for each parameter chosen

error = []

for i in range(len(listPara)):

    # apply ridge regression

    ridgeReg = Ridge(alpha=listPara[i], copy_X=True, fit_intercept=True)

    ridgeReg.fit(xTrain, yTrain)

    optPredict = ridgeReg.predict(xOpt)

    # calculate the error

    error.append(np.sqrt(mean_squared_error(optPredict, yOpt)))

# plot the error

plt.scatter(x=listPara, y=error, color='b', marker='*')

# calculate the best parameter in list

print(pd.Series(data=error, index=listPara).idxmin())
# apply ridge regression

ridge = Ridge(alpha=6.7)

# use all training data available

ridge.fit(xTrainData, dataTrain['SalePrice'])

dataPredicted = ridge.predict(xTestData)

# perform the counter-transformation for log-transformation

dataPredicted = np.exp(list(dataPredicted))-1



idDf = pd.DataFrame(pd.read_csv('../input/test.csv')['Id'])

dataPreDf = pd.DataFrame(dataPredicted, columns=['SalePrice'])

output = pd.concat([idDf, dataPreDf], axis=1)

outputResult = pd.DataFrame(output)

outputResult.to_csv('submission.csv', index=False)
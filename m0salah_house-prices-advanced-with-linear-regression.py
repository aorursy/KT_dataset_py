# Import Library

import numpy as np # linear algebra

import pandas as pd # data processing

from pandas import set_option



import matplotlib

import matplotlib.pyplot as plt

from matplotlib import colors

matplotlib.rcParams['font.size'] = 10

matplotlib.rcParams['figure.figsize'] = (9, 9)

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor



from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
sample_submissionfile = "../input/house-prices-advanced-regression-techniques/sample_submission.csv"

testfile = "../input/house-prices-advanced-regression-techniques/test.csv"

trainfile = "../input/house-prices-advanced-regression-techniques/train.csv"
# Import Dataset

dataset = pd.read_csv(trainfile)

dataset.drop('Id', axis=1, inplace=True)
# Get the Dataset Shape

dataset.shape
# Convert categorical variable into dummy/indicator variables

DatasetDef = dataset.copy()

DatasetDef = pd.get_dummies(dataset)
# Print the shape after Convert categorical variable

DatasetDef.shape
# Compute pairwise correlation of columns

dc = DatasetDef.corr(method='pearson')

print(dc)
# Collect the Features that have correlation higher than 45% or less than -45%

Feature = []

for i in (dc['SalePrice'][dc['SalePrice'] > 0.45]).index:

    Feature.append(i)

for i in (dc['SalePrice'][dc['SalePrice'] < -0.45]).index:

    Feature.append(i)
# Set the SalesPrice as last Feature

Feature.remove('SalePrice')

Feature.append('SalePrice')
# Create a Dataframe with the Highly correlated features

DatasetDef = DatasetDef[Feature]
# Show the Features info

DatasetDef.info()
# we will drop GarageYrBlt as it seem like YearBuilt

DatasetDef.drop('GarageYrBlt', axis=1, inplace=True)
# we will fill the MasVnrArea with the median of the column

median = DatasetDef['MasVnrArea'].median()

DatasetDef['MasVnrArea'].fillna(median, inplace=True)
DatasetDef.info()
DatasetDef['SalePrice'] = np.log(DatasetDef['SalePrice'])

DatasetDef['GrLivArea'] = np.log(DatasetDef['GrLivArea'])
# Correlation Matrix Plot

plt.figure(figsize=(15, 15))

sns.heatmap(DatasetDef.corr(), annot=True)
# we will drop the features with high correlation

DatasetDef.drop(['TotalBsmtSF', '1stFlrSF','GarageArea', 'TotRmsAbvGrd'], axis=1, inplace=True)
# Correlation Matrix Plot

plt.figure(figsize=(15, 15))

sns.heatmap(DatasetDef.corr(), annot=True)
# Feature-Feature Relationships

sns.pairplot(data=DatasetDef)
# Density Plots

DatasetDef.plot(kind='density', subplots=True, layout=(5,5), sharex=False, sharey=False, 

             figsize=(15,15))

plt.show()
# Split-out validation dataset

y = DatasetDef['SalePrice'].values

DatasetDef.drop('SalePrice', axis=1, inplace=True)

X = DatasetDef.values

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=validation_size, random_state=seed)
# Test options and evaluation metric

num_folds = 10

scoring = 'neg_mean_squared_error'
# Spot-Check Algorithms

models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))

models.append(('SVR', SVR()))
# evaluate each model in turn

results = []

names = []

for name, model in models:

    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

    cv_results = -cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# prepare the model

model = LinearRegression()

model.fit(X_train, Y_train)
# transform the validation dataset

predictions = model.predict(X_validation)

print('MSE: {:.3f}'.format(mean_squared_error(Y_validation, predictions)))
# Import Test Feature Dataset

datasettest = pd.read_csv(testfile)

datasettest.drop('Id', axis=1, inplace=True)

datasettest = pd.get_dummies(datasettest)

datasettest = datasettest[Feature[:-1]]

datasettest.drop('GarageYrBlt', axis=1, inplace=True)

datasettest['MasVnrArea'].fillna(0, inplace=True)

datasettest['GarageCars'].fillna(2, inplace=True)

datasettest['GrLivArea'] = np.log(datasettest['GrLivArea'])

datasettest.drop(['TotalBsmtSF', '1stFlrSF','GarageArea', 'TotRmsAbvGrd'], axis=1, inplace=True)

test_X = datasettest[:-1].values
# Import Label Dataset

Label = pd.read_csv(sample_submissionfile)

Label.drop('Id', axis=1, inplace=True)

Label['SalePrice'] = np.log(Label['SalePrice'])

test_Y = Label[:-1].values
test = model.predict(test_X)

print('MSE: {:.4f}'.format(mean_squared_error(test_Y, test)))
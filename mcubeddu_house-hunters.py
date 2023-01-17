# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Plot Style

plt.rcParams.update({'font.size': 14})
# Load train and test data

raw_mydata = pd.read_csv('../input/train.csv', index_col='Id')
# Examine data types

raw_mydata.info()
# Function to transform housing data

def transform_house(data):

    

    # fill nan with 0

    data = data.fillna(0)

    

    # List of categorical, non-numeric variables

    dummy_list = ['MSSubClass', # though numeric in original data, it is categorical

                  'MSZoning', 

                  'Street', 

                  'Alley', 

                  'LotShape', 

                  'LandContour', 

                  'Utilities',

                  'LotConfig',

                  'LandSlope',

                  'Neighborhood',

                  'Condition1',

                  'Condition2',

                  'BldgType',

                  'HouseStyle',

                  'RoofStyle',

                  'RoofMatl',

                  'Exterior1st',

                  'Exterior2nd',

                  'MasVnrType', # Must be used if we use MasVnrArea

                  'ExterQual',

                  'ExterCond',

                  'Foundation',

                  'BsmtQual',

                  'BsmtCond',

                  'BsmtExposure',

                  'BsmtFinType1', # Must be used if we use BsmtFinSF1

                  'BsmtFinType2', # Must be used if we use BsmtFinSF2

                  'Heating',

                  'HeatingQC',

                  'CentralAir',

                  'Electrical',

                  'KitchenQual',

                  'Functional',

                  'FireplaceQu',

                  'GarageType',

                  'GarageFinish',

                  'GarageQual',

                  'GarageCond',

                  'PavedDrive',

                  'PoolQC',

                  'Fence',

                  'MiscFeature',

                  'SaleType',

                  'SaleCondition',

                  'MoSold',

                  #'YrSold', we think we should keep year sold as numeric not categorical

                 ]

    

    # create dummy variables

    for var in dummy_list:

        data = pd.concat([data, pd.get_dummies(data[var], drop_first=True, prefix=var)], axis=1) 

        

    # drop dummy tables

    data = data.drop(dummy_list, axis=1)

    

    # Add total squre foot

    data['TotalSquareFootage'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

    

    return data.copy()
# Transform train and test

mydata = transform_house(raw_mydata)



test, train  = train_test_split(mydata, test_size=0.8, shuffle=True, random_state=8675309)



print(train.shape)

print(test.shape)
train.head()
# Examine data

train.describe()
#Correlation heat map (only integer variables)

corrmat = raw_mydata.iloc[[x - 1 for x in train.index.tolist()]].corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
#Correlation heat map (with descriptive variables converted to dummy variables)

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
plt.scatter(train['GrLivArea'],train['SalePrice'])

plt.xlabel('GrLivArea')

plt.ylabel('SalePrice')

plt.title('Sale Price vs Above grade living area')

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<400000)].index)

test = test.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<400000)].index)
feature = train.dtypes[train.dtypes != "object"].index



# Check the skew

skewed = train[feature].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew: \n")

skewness = pd.DataFrame({'Skew' :skewed})

skewness.head(20)
# Descriptive statistics

train['SalePrice'].describe()
# histogram of Sale Price

sns.distplot(train['SalePrice']);

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



# QQ plot

figure = plt.figure()

r = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
# Transform Sale Price to log(Sale Price)

train["SalePrice"] = np.log(train["SalePrice"])

test["SalePrice"] = np.log(test["SalePrice"])



# Plot the transformed distribution 

sns.distplot(train['SalePrice'] );

plt.ylabel('Frequency')

plt.title('SalePrice distribution')

# Plot the QQ Plot of the transformed variable

figure = plt.figure()

r = stats.probplot(train['SalePrice'], plot=plt)

plt.show()

from pandas import DataFrame

train.to_csv ('prepared_traindata.csv', index=False)

test.to_csv ('prepared_testdata.csv', index=False)
# Scale Train and Test for K-NN model

train_X = train.drop(['SalePrice'], axis=1).copy()

train_Y = np.ravel(train[['SalePrice']])



test_X = test.drop(['SalePrice'], axis=1).copy()

test_Y = np.ravel(test[['SalePrice']])



scaler = StandardScaler()

scaler.fit(train_X)

train_X_scale = scaler.transform(train_X)

test_X_scale = scaler.transform(test_X)



# KNN Grid Search area for best K

k_range = list(range(1,12))

param_grid = dict(n_neighbors = k_range)



# Create model

knn = KNeighborsRegressor()



grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'r2', return_train_score=True)

grid.fit(train_X_scale, train_Y)



print(f'The optimal K-Neighbors is {grid.best_params_}.')
# plot k values

plt.plot(grid.cv_results_['param_n_neighbors'], grid.cv_results_['mean_test_score'])

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-Validated Accuracy')

plt.title('Neighbor CV-Accuracy for KNN')
pd.DataFrame(grid.cv_results_)
print(f'The optimal K-Neighbors is {grid.best_params_}.')
# Predict train_X_scale

knn_yhat = pd.DataFrame(data = grid.predict(test_X_scale),

                    columns = ['knn_yhat'],)

knn_yhat.to_csv('knn_yhat.csv', index=False)
# Define pipeline gor best PCA with KNN

pca = PCA()

knn = KNeighborsRegressor()

k_range = list(range(1,12))

n_components = list(range(1,12))



pipe = Pipeline(steps=[('pca', pca), ('knn', knn)])



# Create Grid to Search

para_grid_pca = {'pca__n_components': n_components,

                'knn__n_neighbors': k_range}



# Create model

grid_pca = GridSearchCV(pipe, para_grid_pca, cv=10, scoring = 'r2', return_train_score=True)

grid_pca.fit(train_X_scale, train_Y)





# plot k values

fig = plt.figure()

fig.set_size_inches(8, 5)

ax = fig.add_subplot(111, projection='3d')

x,y = np.meshgrid(k_range,n_components)

z = np.reshape(grid_pca.cv_results_['mean_test_score'], 

                           (len(k_range),

                            len(n_components)))



ax.plot_wireframe(x,y,z, rstride=2, cstride=2)

ax.set_xlabel('k-neighbors')

ax.set_ylabel('PCA-components')

ax.set_zlabel('Cross-Validated Accuracy')

ax.set_title('CV-Accuracy for KNN-PCA')
pd.DataFrame(grid_pca.cv_results_)
print(f'The optimal PCA-components and K-Neighbors is {grid_pca.best_params_}.')
# Predict train_X_scale

knn_pca_yhat = pd.DataFrame(data = grid_pca.predict(test_X_scale),

                    columns = ['knn_pca_yhat'],)

knn_pca_yhat.to_csv('knn_pca_yhat.csv', index=False)
# Calculate RMSE (Root Mean Squared Error)

np.sqrt(mean_squared_error(test_Y, knn_yhat))
np.sqrt(mean_squared_error(test_Y, knn_pca_yhat))
# Kaggle Kernels are an integrated environment for loading data, running analysis and building 

# your machine learning models. Take some time to explore the UI and interact with it.



# 1- By default, the language selection is set to Python 3. The environment comes preloaded with

#    a ton of libraries. If you are curious about the environment setup, you can take a look 

#    at its the Dockerfile: https://github.com/Kaggle/docker-python/blob/master/Dockerfile



# 2- The Jupyter notebook format allows you to document and execute code alongside. You

#    can execute the code for each blocks by clicking on their play buttons or by running all

#    via the Run menu. 



# 3- You can versionized each step of your kernel by clicking on the Commit button. The right

#    pane allows you to navigate through yo´ur past versions.



# 4- The loaded files for the competition that you participate in will also be found in the right

#    pane via the Workspace component. Explore some of the existing files you have: these 

#    are your tools to solve the current challenge.



# 5- The Setting component in the right pane allows you to switch languages (if you are more 

#    accustomed to R, for example) or to install additional custom libraries, if the one you

#    want

#    to use was not setup in that environment. Again, a lot of libraries are already loaded. 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 

import matplotlib.pyplot as plot



# https://scikit-learn.org/

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import GradientBoostingRegressor
# Encode string fields

def enrichDataset(ds):

  ds['YearBuilt3'] = ds['YearBuilt'].apply(lambda x: x // 3)

  ds['YearBuilt5'] = ds['YearBuilt'].apply(lambda x: x // 5)

  ds['YearBuilt10'] = ds['YearBuilt'].apply(lambda x: x // 10)

  ds['Hood'] = pd.Categorical(ds['Neighborhood']).codes

  ds['LotConfigLabel'] = pd.Categorical(ds['LotConfig']).codes

  ds['LotFrontageLabel'] = pd.Categorical(ds['LotFrontage']).codes

  ds['HouseStyleLabel'] = pd.Categorical(ds['HouseStyle']).codes

  ds['BldgTypeLabel'] = pd.Categorical(ds['BldgType']).codes

  ds['RoofStyleLabel'] = pd.Categorical(ds['RoofStyle']).codes

  ds['FoundationLabel'] = pd.Categorical(ds['Foundation']).codes

  ds['BedroomAbvGrLabel'] = pd.Categorical(ds['BedroomAbvGr']).codes

  ds['GarageCarsLabel'] = pd.Categorical(ds['GarageCars']).codes



# Select our feature(s) from the dataset

def selectFeatures(ds):

  return ds[[

    'YearBuilt5',

    'YrSold',

    'OverallQual',

    'OverallCond',

    'Hood',

    'LotArea',

#     'LotConfigLabel',

#     'LotFrontageLabel',

#     'HouseStyleLabel',

    'FoundationLabel',

    'BedroomAbvGrLabel',

    'GarageCarsLabel'

#     'BldgTypeLabel',

#     'RoofStyleLabel'

]]
# Load the train data set from csv files provided

train_dataset = pd.read_csv('../input/train.csv')



enrichDataset(train_dataset)

datasetX = selectFeatures(train_dataset)

datasetY = train_dataset['SalePrice']

xTrain, xTest, yTrain, yTest = train_test_split(datasetX, datasetY, random_state=0)

model = GradientBoostingRegressor(n_estimators = 100,

                                  max_depth = 3, 

                                  learning_rate = 0.05, 

                                  random_state = 0).fit(xTrain, yTrain)

model.score(xTest, yTest)
# Use the model to predict sales prices on the test dataset

test_dataset = pd.read_csv('../input/test.csv')

enrichDataset(test_dataset)

testFeatures = selectFeatures(test_dataset)

predictions = model.predict(testFeatures)



# Concatenate ids and predictions and save as csv

test_ids = test_dataset[['Id']]

result = pd.concat([

        test_ids,

        pd.DataFrame(data=predictions, columns=['SalePrice'])

    ], axis=1)

result.to_csv("predictions.csv", index=False)

result
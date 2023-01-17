# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure, show



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the train Data

iowa_file_path='/kaggle/input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

home_data.columns
#check which column have a high correlation with SalePrice

figure(figsize=(15,15))

sns.heatmap(home_data.corr())
num_correlation = home_data.select_dtypes(exclude='object').corr()

corr = num_correlation.corr()

print(corr['SalePrice'].sort_values(ascending=False))
# Create target object and call it y

y = home_data.SalePrice

# Create X

#features = ['OverallQual','LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF','FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GrLivArea','GarageCars', 'GarageArea']

featurestop=['OverallQual','TotalBsmtSF', 'YearBuilt','YearRemodAdd','GarageYrBlt','Fireplaces', '1stFlrSF', 'MasVnrArea','FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GrLivArea','GarageCars', 'GarageArea']

X = home_data[featurestop]

home_data[featurestop]

sns.heatmap(X.isnull(),yticklabels=False, cbar=False, cmap='viridis')
GarageYrBltmean=X.loc[:,"GarageYrBlt"].mean()

MasVnrAreamean=X.loc[:,"MasVnrArea"].mean()

print(GarageYrBltmean,MasVnrAreamean)
X['GarageYrBlt'].fillna(GarageYrBltmean,inplace = True)

X['MasVnrArea'].fillna(MasVnrAreamean,inplace = True)
# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
##Check TestData

# path to file you will use for predictions

test_data_path = '/kaggle/input/home-data-for-ml-course/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[featurestop]

#test_X.dropna(inplace=True)

test_X.info()
#working with missing Values

GarageCarsmean=test_X.loc[:,"GarageCars"].mean()

GarageAreamean=test_X.loc[:,"GarageArea"].mean()

GarageYrBltmean=test_X.loc[:,"GarageYrBlt"].mean()

MasVnrAreamean=test_X.loc[:,"MasVnrArea"].mean()

TotalBsmtSFmean=test_X.loc[:,"TotalBsmtSF"].mean()

print(GarageYrBltmean,MasVnrAreamean)

print(GarageCarsmean,GarageAreamean)
test_X['GarageArea'].fillna(GarageAreamean,inplace = True)

test_X['GarageYrBlt'].fillna(GarageYrBltmean,inplace = True)

test_X['MasVnrArea'].fillna(MasVnrAreamean,inplace = True)

test_X['GarageCars'].fillna(GarageCarsmean,inplace = True)

test_X['TotalBsmtSF'].fillna(TotalBsmtSFmean,inplace = True)

test_X.info()
rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X, y)
# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': test_preds})

#output.to_csv('submission.csv', index=False)

rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X, y)



# Then in last code cell





output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
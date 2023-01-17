import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from sklearn.ensemble import RandomForestRegressor

train = '../input/train.csv'
test  =  '../input/test.csv'
train = pd.read_csv(train)
test = pd.read_csv(test)


#missing data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)



# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['OverallQual', 'GrLivArea', 'GarageCars','GarageArea', 'TotalBsmtSF', '1stFlrSF']


# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
#missing data test
total_test = test.isnull().sum().sort_values(ascending=False)
percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data_test.head(20)

test.shape
test = test.drop((missing_data[missing_data['Total'] > 1]).index,1)

predict_test = ['OverallQual', 'GrLivArea', 'GarageCars','GarageArea', 'TotalBsmtSF', '1stFlrSF']

test.fillna(0, inplace=True)


test.isnull().sum().max() #just checking that there's no missing data missing...


# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predict_test]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission

# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

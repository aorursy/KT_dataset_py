# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn import preprocessing

from sklearn.model_selection import train_test_split





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#TFI (tab food investments) has provided a dataset with 137 restaurants in the training set, and a test set of 100000 restaurants..

train =  pd.read_csv('../input/restaurant-revenue-prediction/train.csv')

test = pd.read_csv('../input/restaurant-revenue-prediction/test.csv')
train['Days_Open'] = (pd.to_datetime('2015-03-23') - pd.to_datetime(train['Open Date'])).dt.days

test['Days_Open'] = (pd.to_datetime('2015-03-23') - pd.to_datetime(test['Open Date'])).dt.days

train = train.drop('Open Date', axis=1)

test = test.drop('Open Date', axis=1)



train['Type_IL'] = np.where(train['Type'] == 'IL', 1, 0)

train['Type_FC'] = np.where(train['Type'] == 'FC', 1, 0)

train['Type_DT'] = np.where(train['Type'] == 'DT', 1, 0)

test['Type_IL'] = np.where(test['Type'] == 'IL', 1, 0)

test['Type_FC'] = np.where(test['Type'] == 'FC', 1, 0)

test['Type_DT'] = np.where(test['Type'] == 'DT', 1, 0)

train = train.drop('Type', axis=1)

test = test.drop('Type', axis=1)
y= train.revenue

x_train= train.drop(['Id', 'revenue'], axis=1)

x_test = test.drop(['Id'], axis=1)
from sklearn.preprocessing import LabelEncoder

# Processing the categorical columns to provide vector form of feature

class DataFrameProcess:

    def __init__(self,df,col):

        self.df =df

        self.col=col

    def dataEncoding(self):

        if self.df[self.col].dtype.name == 'object' or self.df[self.col].dtype.name == 'category':

            le = LabelEncoder()

            self.df[self.col] = le.fit_transform(self.df[self.col])    





def data_transform(df):  

    for col in df.columns:

        data_prcs = DataFrameProcess(df,col)

        data_prcs.dataEncoding()  

data_transform(x_train) 

data_transform(x_test)
x_train.shape
y.shape
#newdf_train = pd.DataFrame(np.repeat(x_train.values,100000,axis=0))

#newdf_train.columns = x_train.columns



#newdf_y = pd.DataFrame(np.repeat(y.values,100000,axis=0))

#newdf_y.columns = y.columns

#print(newdf_y)
#X_train, c, y_train, y_valid = train_test_split(newdf_train, newdf_y, train_size=0.8, test_size=0.2, random_state=0)
#X_valid.shape
#X_train.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt



rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(x_train, y)

rf_predictions = rf_model.predict(x_test)
test_label=pd.read_csv('../input/restaurant-revenue-prediction/sampleSubmission.csv')  # test target

from sklearn.metrics import mean_squared_error

from math import sqrt

label_list=test_label['Prediction'].tolist()
RandomForestRegressor_RMSE= sqrt(mean_squared_error(label_list, rf_predictions))

print('Root Mean squared error {}'.format(RandomForestRegressor_RMSE))

submission = pd.DataFrame({

        "Id": test["Id"],

        "Prediction": rf_predictions

    })

submission.to_csv('submission.csv',header=True, index=False)

print('done')
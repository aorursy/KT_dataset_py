# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Converting the given CSV files into Pandas Dataframe format so that we can start working with them

df = pd.read_csv('/kaggle/input/wec-intelli-atreya/Train.csv')

df_test = pd.read_csv('/kaggle/input/wec-intelli-atreya/Test_data.csv') 

df_op = pd.read_csv('/kaggle/input/wec-intelli-atreya/output.csv')
#Just cross checking :D

import os

print((os.listdir('../input/')))
#Train data samples

df.head()

#Test data samples

df_test.head()
df.drop(['F1', 'F2'], axis = 1, inplace = True) #Dropping indexing columns
df.head() #Checking that the said columns have been dropped
train_x = df.loc[:, 'F3':'F17']  #Splitting train DF into input and output values

train_y = df.loc[:, 'O/P']

test_x = df_test.loc[:, 'F3':'F17'] #Reading all the test DF values
#Importing stuff that might be needed and will be used

import xgboost

from xgboost import plot_importance

import matplotlib.pyplot as plt

import seaborn

from sklearn.model_selection import cross_val_score,KFold

from sklearn.metrics import mean_absolute_error

test_index=df_test['Unnamed: 0']  
#Using XGB since RF was used in the baseline :P

xgb = xgboost.XGBRegressor(n_estimators=3300, learning_rate=0.01, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)
xgb.fit(train_x,train_y) #Training the model
df_test.head() #Checking the test DF
scores = cross_val_score(xgb, train_x, train_y, scoring="neg_mean_squared_error", cv=5)  #Metric Check
print(scores)
pred = xgb.predict(test_x) #Making the actual regression predictions now
print(pred)
df_op.head()  #Checking output data format
#Writing the predicted results

result2=pd.DataFrame()

result2['Id'] = test_index

result2['PredictedValue'] = pd.DataFrame(pred)

result2.head()
result2.to_csv('op_Atreya4.csv', index=False)  #DF to CSV conversion
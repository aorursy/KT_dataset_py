# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/data.csv')
#Cleaning the dataset

#First look dataset Column

df.info()

#Lets Check out the nul value and dealing with them

df.isnull().sum()
import numpy as np 

import pandas as pd
df = pd.read_csv('../input/data.csv')
#Cleaning the dataset

#First look dataset Column

df.info()

#Lets Check out the nul value and dealing with them

df.isnull().sum()

#Now convert vedio upload, Subscribers and Video views to the numeric value 

df['Video Uploads'] = df['Video Uploads'].apply(pd.to_numeric, errors = 'coerce')

df['Subscribers'] = df['Subscribers'].apply(pd.to_numeric, errors = 'coerce')

df['Video views'] = df['Video views'].apply(pd.to_numeric, errors = 'coerce')

#Now lets take care of missing data (In this case replace null value with the mean of the prticula column)

df['Video Uploads'] = df['Video Uploads'].fillna(df['Video Uploads'].mean())

df['Subscribers'] = df['Subscribers'].fillna(df['Subscribers'].mean())
#Now build the machine learning model

#Now lets take our matrics of features

x_trail = df[['Video Uploads', 'Video views']]

x = x_trail.iloc[:,:].values

y = df.iloc[:, 4].values
#Now split the data into training and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Fitting Linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
#predicting test set result

y_pred = regressor.predict(x_test)



#Building the Optimal Model using Backward Elimination

import statsmodels.formula.api as sm

x_opt = x[:, [0,1,]]

regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()

regressor_OLS.summary()
#Visualising the Predicted Result

import matplotlib.pyplot as plt

plt.plot(y_test, color = 'red', label = 'Actual')

plt.plot(y_pred, color = 'blue', label = 'Predicted')

plt.xlabel('Number of Reviews')

plt.ylabel('Rating')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()



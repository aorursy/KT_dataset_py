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
import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense

import pandas as pd

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from datetime import datetime
df=pd.read_csv('../input/nifty-indices-dataset/INDIAVIX.csv')

df.head()
df.isnull().sum()

df.shape
df.columns
df=df[ ['Close','Change','%Change'] ]
df['FuturePrice']=df['Close'].shift(-7)# predecting last 7 values for future price
features=np.array(df.drop(['FuturePrice'],1))

features=preprocessing.scale(features)
features = features[:2706]        

predictions = features[-7:] # for future predictions

labels=np.array(df['FuturePrice'])

labels=labels[:2706]
print(len(features))

print(len(predictions))

print(len(labels))

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size=0.2,random_state=10)



print(train_features.shape)

print(test_features.shape)

print(train_labels.shape)

print(test_labels.shape)
model=LinearRegression()

model.fit(train_features,train_labels)

accuracy=model.score(test_features,test_labels)

print(accuracy)
import xgboost as xgb
model1 = xgb.XGBRegressor()

model1.fit(train_features,train_labels)

accuracy1=model1.score(test_features,test_labels)

print(accuracy1)
from sklearn.svm import SVR
model2=SVR(epsilon=0.2)

model2.fit(train_features,train_labels)

accuracy2=model2.score(test_features,test_labels)

print(accuracy2)
# best accuracy by Linear Regression
prediction_prices=model.predict(predictions)

print(prediction_prices)
for index in range(-7,0,1):

    rowIndex = df.iloc[-index].name

    df['FuturePrice'][index]=prediction_prices[index+7]
df=df.truncate(before='2020-04-29')

df['FuturePrice'].plot()

df['Close'].plot()

plt.legend()

plt.xlabel('Date')

plt.ylabel('Price')

plt.show()

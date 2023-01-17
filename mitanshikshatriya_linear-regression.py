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
filename = '/kaggle/input/headbrain/headbrain.csv'

df = pd.read_csv(filename)

df.head()
df.shape
#checking for null values

df.isnull().sum()

#eda

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.regplot(x=df['Head Size(cm^3)'],y=df['Brain Weight(grams)'])
sns.lmplot(x='Head Size(cm^3)',y='Brain Weight(grams)',hue='Gender',data=df)
sns.lmplot(x='Head Size(cm^3)',y='Brain Weight(grams)',hue='Age Range',data=df)
#using head size for predicting brain size by applying linear regression from scratch
# y = mx+ c

X = df['Head Size(cm^3)'].values

y = df['Brain Weight(grams)'].values



#mean

X_mean = np.mean(X)

y_mean = np.mean(y)



#length

n = len(X)



numerator=0

denominator=0



for i in range(n):

    numerator += (X[i]-X_mean)*(y[i]-y_mean)

    denominator += (X[i]-X_mean)**2

    

m = numerator/denominator

c = y_mean - (m*X_mean)



print(m,c)
#predicting brain size

#brain wiegts = m*(head size) + c

y_preds=[]

for i in range(n):

    y_preds.append(m*X[i]+c)

    

y_preds
#measuring accuracy using root mean squared error

rsme=0

for i in range(n):

    rsme+=(y[i]-y_preds[i])**2

    

rsme=np.sqrt(rsme/n)

rsme
#using sklearn for linear regression

#X.shape

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



X = df['Head Size(cm^3)']

y = df['Brain Weight(grams)']



X=np.array(X).reshape(-1,1)

#X.shape



reg = LinearRegression()

reg.fit(X,y)



y_pred = reg.predict(X)

y_pred
#calculating rsme

rsme = np.sqrt(mean_squared_error(y,y_pred))

rsme
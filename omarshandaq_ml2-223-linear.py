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
df=pd.read_csv("/kaggle/input/housesalesprediction/kc_house_data.csv", header=0)

df.head()
from sklearn import linear_model

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import sklearn.metrics as met

import datetime
#Data Preprocessing

current_year = datetime.datetime.now().year

df["age_of_house"] = current_year - pd.to_datetime(df["date"]).dt.year

dates=df['date']

x=0

for i in dates:

    j=i[:8]

    df['date'][x]=j

    x += 1

df.head()
#Data Preprocessing

passing_columns = ['price', 'bedrooms', 'bathrooms', 'date','grade','sqft_lot','yr_built','age_of_house', 'sqft_living15']#, 'condition',

       #, 'sqft_basement', 'yr_renovated',

X=df[passing_columns]

y=df['sqft_living']

X.head(6)
#Train & test data

lr = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#Model training

lr.fit(X_train,y_train)



#Model testing

z=0

y_pred = lr.predict(X)

print("Sample"'\n')

for i in y_pred:

    if z <= 15:

        print("predicted value:", i, "comparing to real value:", y[z])

    z += 1
#Evaluation 

count=0

for i in passing_columns:

    print("the slop of",passing_columns[count], "=" , lr.coef_[count])

    count +=1

accuracy = lr.score(X_test, y_test)

r_sequare = lr.score(X_train,y_train)



print('\n'"Accuracy: {}%".format(int(round(accuracy * 100))))

print('\n'"coefficiant of determination:", r_sequare)

print('\n'"model interseption:", lr.intercept_)

print('\n''Mean squared error: %.2f' % met.mean_squared_error(y, y_pred))

#output plots "sqft_living" with each column

for plot in passing_columns:

    plt.scatter(X[plot], y)

    sns.scatterplot(X[plot],y_pred,color="red")

    plt.show()
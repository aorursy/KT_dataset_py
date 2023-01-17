# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('/kaggle/input/boston-housepredict/boston_train.csv')
df_train.head()
# Correlation

plt.subplots(figsize=(20,15))

correlation_matrix = df_train.corr().round(2)

sns_plot=sns.heatmap(data=correlation_matrix, annot=True)
price=df_train['medv']

data=df_train.drop(['medv','nox','chas','zn','indus','rad','tax','dis','age','ID','crim','black','ptratio'],axis=1)
data.head()
plt.figure(figsize=(20,5))

features =['lstat','rm']

target = price

for i, col in enumerate(features):

    plt.subplot(1,len(features),i+1)

    x = data[col]

    y=target

    plt.scatter(x,y,marker='o')

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel('Price')
print("BASIC STATS FOR OUR THE BOSTON HOUSING DATASET  \n")

MAX_PRICE=np.max(price)

MIN_PRICE=np.min(price)

MEAN_PRICE=np.mean(price)

MEDIAN_PRICE=np.median(price)

STD_PRICE=np.std(price)

print("Max Price in USD 1000’s = ${:,.2f}".format(MAX_PRICE))

print("Min Price in USD 1000’s = ${:,.2f}".format(MIN_PRICE))

print("Mean Price in USD 1000’s = ${:,.2f}".format(MEAN_PRICE))

print("Median Price in USD 1000’s = ${:,.2f}".format(MEDIAN_PRICE))

print("Standard Dev Price in USD 1000’s = ${:,.2f}".format(STD_PRICE))
from sklearn.model_selection import train_test_split



# TODO: Shuffle and split the data into training and testing subsets

X_train, X_test, y_train, y_test = train_test_split(data, price, random_state=0, test_size=0.20)



# Success

print ("Training and testing split was successful.")



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

model = LinearRegression().fit(X_train,y_train)

train_pred = model.predict(X_test)

r_sq = model.score(X_train,y_train)

print('The model performance for training set')

print('---------------------------------------')

print('coefficient of determination:',r_sq)

print('Mean Absolute Error:',mean_absolute_error(y_test,train_pred))

print('Root Mean Squared Error:',np.sqrt

      (mean_absolute_error(y_test,train_pred)))



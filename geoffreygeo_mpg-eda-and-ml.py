# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df =pd.read_csv('../input/auto-mpg.csv')

df = df[df.horsepower != '?']

df.head()
unique_car_names = df['car name'].unique()

print(unique_car_names)
sns.countplot(df['cylinders'])
sns.countplot(df['origin'])

def catg_origin(df):

    if(df['origin']==1):

        return 'USA'

    elif(df['origin']==2):

        return 'Europe'

    else:

        return 'Japan'

df['origin'] = df.apply(catg_origin,axis=1)

#One hot encoding for origin 

one_hot_origin = pd.get_dummies(df.origin, prefix='origin')

one_hot_origin.head()

df = df.join(one_hot_origin)

df.pop('origin')

df.shape
sns.countplot(df['model year'])
df.drop('car name', axis=1, inplace=True)

df.head()
#Spliting the Training and Testing dataset

trainSet, testSet = train_test_split(df, test_size = 0.33)

trainLabel = trainSet.pop('mpg')

testLabel = testSet.pop('mpg')

print("TestSet {}\nTrainSet {}".format(trainSet.shape,testSet.shape))
df.isnull().sum()
#Defining Machine learning model 

linear=LinearRegression()

linear.fit(trainSet,trainLabel)
predicted_values = linear.predict(testSet)
# Model Output

# a. Coefficient — the slop of the line

print("Coefficients(slope of the line): {}".format(linear.coef_))

# b. the error — the mean square error

display("Mean squared Error - {}".format(mean_squared_error(testLabel,predicted_values)))

# c. R-square — how well x accout for the varaince of Y

print("R-Square : {}".format(r2_score(testLabel,predicted_values)))



fig,ax = plt.subplots()

ax.scatter(testLabel, predicted_values)

ax.plot([testLabel.min(), testLabel.max()], [testLabel.min(), testLabel.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

fig.show()
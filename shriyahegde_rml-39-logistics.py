# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/glass/glass.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Step 1: Import Library

# Import data analysis modules

import numpy as np

import pandas as pd

import os

# to save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#Step 2 : Data import

# Use pandas to read in csv file



train = pd.read_csv('/kaggle/input/glass/glass.csv')

#this is just a comment

train.head(5)
train.describe()
#Step 3: Clean up data

# Use the .isnull() method to locate missing data

missing_values = train.isnull()

missing_values.head(5)
sns.scatterplot(x= 'Na', y='Type', data = train.tail(15000))
X = train[['Al']]

y = train['Type']
from sklearn.model_selection import train_test_split

# Split data set into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
print(X_train.shape)

print(y_test.shape)

print(y_train.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

output_model=model.fit(X_train, y_train)

output_model
pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': Ypredict.flatten()})

df
#Understanding accuracy

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



predictions = model.predict(X_test)

#print("",classification_report(y_test, predictions))

#print("confusion_matrix",confusion_matrix(y_test, predictions))

#print("accuracy_score",accuracy_score(y_test, predictions))

##**Accuracy is a classification metric. You can't use it with a regression. See the documentation for info on the various metrics.

#For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).

#print("Score",score(y_test, X_test))

#score(self, X, y, sample_weight=None)

## setting plot style 

plt.style.use('fivethirtyeight') 

  

## plotting residual errors in training data 

plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, 

            color = "green", s = 1, label = 'Train data' ,linewidth = 5) 

  

## plotting residual errors in test data 

plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, 

            color = "blue", s = 1, label = 'Test data' ,linewidth = 4) 

  

## plotting line for zero residual error 

plt.hlines(y = 0, xmin = 0, xmax = 4, linewidth = 2) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 
# plotting regression line

ax = plt.axes()

ax.scatter(X['Al'], y)

plt.title("Input Data and regression line ") 

ax.plot(X_test, Ypredict, color ='Red')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.axis('tight')

plt.show()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Ypredict))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, Ypredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Ypredict)))


print("confusion_matrix",confusion_matrix(y_test, predictions))
print("",classification_report(y_test, predictions))
print("accuracy_score",accuracy_score(y_test, predictions))
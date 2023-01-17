# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/kc_house_data.csv'):

    for filename in filenames:

        print(os.path.join(dirname, kc_house_data.csv))



# Any results you write to the current directory are saved as output.
#Step 2 : Data import

# Use pandas to read in csv file

#os.chdir(r'C:\Users\mndesh\Documents')

#os.chdir('C:\\Users\\mndesh\\Documents')

os.path.isfile('/kaggle/input/kc_house_data.csv')

#train = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

#this is just a comment

#train.head(5)
df = pd.read_csv('/kaggle/input/kc_house_data.csv')

df.head(5)
df.describe()
df.dtypes
missing_values = df.isnull()

missing_values.head(5)
y = df['sqft_living']
x = df['sqft_living15']
import seaborn as sns
sns.scatterplot(x,y)
#Step 5.1: Prepare input X parameters/features and output y



# Split data into 'X' features and 'y' target label sets

x = df[['sqft_living15']]

y = df['sqft_living']
#Step 5.2: Split data into Train and test



# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
print(x.shape)

#print(x_test.head())

#print(y_train.head())

print(y.shape)
import numpy as ny
#Step 6 : Run the model



# Import model for fitting

from sklearn.linear_model import LinearRegression

# Create instance (i.e. object) of LogisticRegression

#model = LogisticRegression()



#You can try follwoing variation on above model, above is just default one

model = LinearRegression()

# Fit the model using the training data

# X_train -> parameter supplies the data features

# y_train -> parameter supplies the target labels

output_model=model.fit(x,y)

#output =x_test

#output['vehicleTypeId'] = y_test

output_model
import pickle
#Step 7.0 Save the model in pickle

#Save to file in the current working directory

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(x_test, y_test)

#print(score)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': Ypredict.flatten()})

df
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Ypredict))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, Ypredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Ypredict)))
import matplotlib.pyplot as plt
# plotting regression line

ax = plt.axes()

ax.scatter(x, y)

plt.title("Input Data and regression line ") 

ax.plot(x_test, Ypredict, color ='Red')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.axis('tight')

plt.show()
#Step 7.1: Understanding accuracy

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



predictions = model.predict(x_test)

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

plt.scatter(model.predict(x_train), model.predict(x_train) - y_train, 

            color = "green", s = 1, label = 'Train data' ,linewidth = 5) 

  

## plotting residual errors in test data 

plt.scatter(model.predict(x_test), model.predict(x_test) - y_test, 

            color = "blue", s = 1, label = 'Test data' ,linewidth = 4) 

  

## plotting line for zero residual error 

plt.hlines(y = 0, xmin = 0, xmax = 4, linewidth = 2) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 

os.path.isfile('/kaggle/input/kc_house_data.csv')

check = pd.read_csv('/kaggle/input/kc_house_data.csv')

print('Importing data to solve for')

print(df.head(5))
Ypredict = pickle_model.predict(x_test)

#Ypredict = pickle_model.predict(check[['x']])

output=x_test

output['y_Predicted']=Ypredict



print(output)

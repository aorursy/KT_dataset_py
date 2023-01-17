# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

input_file = ""

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        input_file = os.path.join(dirname, filename)

df = pd.read_csv(input_file)

df.head(5)

# Any results you write to the current directory are saved as output.
df.head(5)
df.info()
df.describe()
df.columns
missing_values = df.isnull()

missing_values.head(5)

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
X = df[['sqft_basement', 'sqft_above']]

y = df['sqft_living']
X.head(5)
X.shape
y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)
print(X_train.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

output_model = lm.fit(X_train,y_train)

output_model
output_model.intercept_
output_model.coef_
X_train.columns
cdf = pd.DataFrame(output_model.coef_,X_train.columns, columns=['Coeff'])
cdf
predictions = output_model.predict(X_test)

predictions
plt.scatter(y_test,predictions)
sns.distplot(y_test-predictions)
from sklearn import metrics

metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))
import pickle

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(lm, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(X_test, y_test)

print("Test score: {0:.2f} %".format(100 * score))

Ypredict = pickle_model.predict(X_test)
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': Ypredict.flatten()})

df1
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



predictions = lm.predict(X_test)

plt.style.use('fivethirtyeight') 

plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, 

            color = "green", s = 1, label = 'Train data' ,linewidth = 5) 

plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, 

            color = "blue", s = 1, label = 'Test data' ,linewidth = 4) 

plt.hlines(y = 0, xmin = 0, xmax = 4, linewidth = 2) 

plt.legend(loc = 'upper right') 

plt.title("Residual errors") 

plt.show() 
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Ypredict))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, Ypredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Ypredict)))
ax = plt.axes()

ax.scatter(df['sqft_above'], y)

plt.title("Input Data and regression line ") 

ax.plot(X_test, Ypredict, color ='Red')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.axis('tight')

plt.show()
df2 = df[(df['price'] > 300000) & (df['price'] < 600000)]

df2.head(5)
Ypredict = pickle_model.predict(df2[['sqft_basement', 'sqft_above']])

output=df2[['sqft_basement', 'sqft_above']]

output['Y_Predicted']=Ypredict

print(output)

print( 'Saved with Name: Output_Test_Data.csv')

output.to_csv( 'Output_Test_Data.csv')
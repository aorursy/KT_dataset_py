# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Loading the data

df = pd.read_csv('/kaggle/input/tv-marketing-data/tvmarketing.csv')
# Let's check the length of the data

len(df)
# Check out the head of the dataframe

df.head()
# Let's check stats of the model

df.describe()
# Now let's try to make a scatter plot of the data

sns.scatterplot(df['TV'],df['Sales'])
# Let's check if we have any missing value in the data

df.isnull().sum()
# Now divide the data into depenedent(output) and independent(features/input) variables

# X-----> feature     &      y------>output



X = df.iloc[:,0] 

y = df.iloc[:,1]
# First of all split the data into train and test set

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 56)
X_train.shape
X_train = X_train.values.reshape(-1,1)

y_train = y_train.values.reshape(-1,1)

X_test = X_test.values.reshape(-1,1)

y_test = y_test.values.reshape(-1,1)
# Lets's import the linear regression algorithm

from sklearn.linear_model import LinearRegression



lr = LinearRegression()
# Let's make the linear model by fitting on to the data

model = lr.fit(X_train, y_train)
# Let's predict

y_pred = model.predict(X_test)
# Let's compare predicted and actual values

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df

# Let's plot these results

plt.scatter(X_test, y_test,  color='blue')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()
# Let's check our model's performance in terms of different metrices

from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
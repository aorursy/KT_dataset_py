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
df_2013 = pd.read_csv("/kaggle/input/air-quality/real_2013.csv")
df_2013.head()
df_2014 = pd.read_csv("/kaggle/input/air-quality/real_2014.csv")
df_2014.head()
df_2015 = pd.read_csv("/kaggle/input/air-quality/real_2015.csv")
df_2015.shape
df_2015.head()
frames = [df_2013,df_2014,df_2015]

data = pd.concat(frames)
data.shape
missing_val_count_by_column = (data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
data2 = data.dropna(axis=0) # Since only one row missing
data2.shape
import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
X = data2[['T','TM','Tm','SLP','H','VV','V','VM']]

y = data2['PM 2.5']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  

regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:

print(regressor.intercept_)

#For retrieving the slope:

print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
# Plot outputs

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='black')

plt.xticks(())

plt.yticks(())



plt.show()
test = pd.read_csv("/kaggle/input/air-quality/real_2016.csv")
test.head()
X_2016 = test[['T','TM','Tm','SLP','H','VV','V','VM']]

y_2016 = test['PM 2.5']
y_pred2 = regressor.predict(X_2016)
df = pd.DataFrame({'Actual': y_2016, 'Predicted': y_pred2})

df
from sklearn.metrics import mean_squared_error,r2_score

mean_squared_error(y_2016, y_pred2)
r2_score(y_2016, y_pred2)
# Plot outputs

import matplotlib.pyplot as plt

plt.scatter(y_2016, y_pred2, color='black')

plt.xticks(())

plt.yticks(())



plt.show()
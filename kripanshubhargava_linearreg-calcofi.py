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
# helper libraries
import pandas as pd
import numpy as np
# data visualization lib
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#machine learning sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# read csv
bottle_csv = pd.read_csv("/kaggle/input/calcofi/bottle.csv")
# convert into dataframe
bottle_df = pd.DataFrame(bottle_csv)
#shape of the dataset
bottle_df.shape
# see the first 5 rows of the df
bottle_df.head(5)
#get info about the data frame
bottle_df.info()
# describe the dataset
bottle_df.describe()
# total null values percentage 
bottle_df.isnull().sum()

# Scatter plot chart 
plt.figure(figsize=(12, 7))
sns.scatterplot(x=bottle_df['T_degC'], y=bottle_df['Salnty'])
new_df = bottle_df[["Salnty", "T_degC"]]
new_df["T_degC"] = new_df["T_degC"].fillna(new_df["T_degC"].mean())
new_df["Salnty"] = new_df["Salnty"].fillna(new_df["Salnty"].mean())
new_df.isnull().sum()
new_df.head()
new_df.describe()
# 
sns.pairplot(new_df)
X = np.array(new_df["Salnty"]).reshape(-1, 1)
y = np.array(new_df["T_degC"]).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fit the model
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print(regressor.intercept_)#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)
accuracy = regressor.score(X_test, y_test)
print(accuracy)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
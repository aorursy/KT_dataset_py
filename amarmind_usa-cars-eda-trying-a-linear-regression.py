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
df = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")
df.head()
df.tail()
df = df.drop(['country', 'condition'], axis = 1)
df.head()
df.shape

duplicate_rows_df = df[df.duplicated()]
duplicate_rows_df.shape
df.count()
df.describe()
df[df['price']==0].count()
import seaborn as sns
sns.set(color_codes=True)
sns.boxplot(x=df['price'])
df[df['mileage']==0].count()
sns.boxplot(x=df['mileage'])
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df = df[~((df < (Q1-1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape
df[df['price']==0].count()
df[df['mileage']==0].count()
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
# Plotting a Histogram
df['brand'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by brand")
plt.ylabel("Number of cars")
plt.xlabel("Brand");
df['state'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by state")
plt.ylabel("Number of cars")
plt.xlabel("State");
df['color'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by color")
plt.ylabel("Number of cars")
plt.xlabel("Color");
df['year'].value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of cars by Year")
plt.ylabel("Number of cars")
plt.xlabel("Year");
# Finding the relations between the variables
plt.figure(figsize=(20,10))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c
# Plotting a scatter plot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['year'], df['price'])
ax.set_xlabel('Year')
ax.set_ylabel('Price')
plt.show()
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(df['mileage'], df['price'])
ax.set_xlabel('Mileage')
ax.set_ylabel('Price')
plt.show()
x= np.array(df['mileage'])
y= np.array(df['price'])
plt.plot(x, y, 'o')
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

X = df['mileage'].values.reshape(-1,1)
y = df['price'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
df_test = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_test
df1 = df_test.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('10% of Mean Price:', df['price'].mean() * 0.1)
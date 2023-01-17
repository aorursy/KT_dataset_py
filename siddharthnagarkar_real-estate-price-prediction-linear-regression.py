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
import matplotlib.pyplot as plt
import seaborn as sns
# Loading dataset
df = pd.read_csv('../input/real-estate/real_estate.csv')

# Viewing dataset
df.head()
# Checking statistics
df.describe()
# Checking co-related variables
sns.pairplot(df)
plt.show()
# Plotting 'view' and 'year' with respect to 'price'
plt.figure(figsize=(10, 7))
plt.subplot(1,2,1)
sns.boxplot(x = 'view', y = 'price', data = df)
plt.subplot(1,2,2)
sns.boxplot(x = 'year', y = 'price', data = df)
plt.tight_layout()
plt.show()
# Mapping 'view' variable

df['view']=df['view'].map({'No sea view':1,'Sea view':0})
df.head()
# Creating dummy variable for 'year'

df['year'].value_counts()
year = pd.get_dummies(df['year'], drop_first = True)
year.head()
# Concatenating year and df dataframes

df = pd.concat([df, year], axis = 1)
df.head()
# Dropping 'year' column

df.drop(columns = ['year'], inplace = True)
df.head()
# Spliting data into Train and Test using sklearn library

from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(df, train_size = 0.7, test_size = 0.3, random_state = 100)
# Using MinMaxScaler to scaler numerical variables

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num = ['price','size']

#Since, we want to learn the Min and Max values from the train dataset, hence we'll use fit_transform

df_train[num] = scaler.fit_transform(df_train[num])
df_train.head()
# Observing co-related variables

plt.figure(figsize=[12,12])
sns.heatmap(df_train.corr(), annot = True, cmap="RdYlBu_r")
plt.show()
# Using regplot to observe regression line

plt.figure(figsize=[10,10])
sns.regplot(x = 'size', y = 'price', data = df_train)
plt.show()
# Creating X_train and y_train

y_train = df_train.pop('price')
X_train = df_train
# Using statsmodel for model building

import statsmodels.api as sm

# Adding constant to X_train
X_train_lm = sm.add_constant(X_train)

# Fitting first model
lr= sm.OLS(y_train, X_train_lm).fit()

# Checking summary 
print(lr.summary())
# Using VIF to eliminate variables

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# For variable '2009' - VIF < 5 OK and P-value > 0.05 Not OK, hence, dropping '2009'

X = X_train.drop(2009, 1,)
# Adding constant to X_train

X_train_lm = sm.add_constant(X)

# Fitting second model

lr_2 = sm.OLS(y_train, X_train_lm).fit()

# Checking summary 
print(lr_2.summary())
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Checking error terms using predicted y_train_pred

y_train_pred = lr_2.predict(X_train_lm)

fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  
plt.xlabel('Errors', fontsize = 18)
plt.show()
num = ['price','size']

# Transforming test dataset with learned MIN & MAX values

df_test[num] = scaler.fit_transform(df_test[num])
df_test = df_test.drop(2009, 1,)
df_test.head()
# Creating X_test and y_test

y_test = df_test.pop('price')
X_test = df_test
# Adding constant

X_test_m2 = sm.add_constant(X_test)
# Predicting

y_pred_m2 = lr_2.predict(X_test_m2)
fig = plt.figure()
sns.regplot(y_test, y_pred_m2)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)   
plt.show()
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_m2))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_m2))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_m2)))
#Observing actual and predicted values, using stacked bar graph

df_1 = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred_m2.values.flatten()})

df_2 = df_1.head(30)
df_2.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

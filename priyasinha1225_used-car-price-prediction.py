# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import r2_score
import re
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/used-cars-price-prediction/train-data.csv')
df_test = pd.read_csv('/kaggle/input/used-cars-price-prediction/test-data.csv')
df_train.head()
df_train.info()
df_train.shape
df_train.isnull().sum()
df_train = df_train.rename(columns = {'Unnamed: 0': 'id'})
df_train["Seats"].fillna(value = 5.0, inplace=True)
df_train.Seats[df_train.Seats == 0.0] = 5.0
df_train.isna().sum()

df_train.Mileage[df_train.Mileage == '0.0 kmpl'] = np.nan
df_train['Mileage'] = df_train['Mileage'].apply(lambda x: re.sub(r'(\d+\.\d+)\s(kmpl|km\/kg)', 
                                                                 r'\1', str(x)))
df_train['Mileage'] = df_train['Mileage'].astype(float)
df_train['Mileage'].mode()
df_train['Mileage'].fillna(value = 17.0, inplace = True)
df_train.isna().sum()

df_train['Engine'] = df_train['Engine'].apply(lambda x: re.sub(r'(\d+)\s(CC)', r'\1', str(x)))
df_train['Engine'] = df_train['Engine'].astype(float)
df_train['Engine'].mode()
df_train['Engine'].fillna(value = 1197.0, inplace = True)
df_train.isna().sum()

df_train['Power'] = df_train['Power'].str.split(' ').str[0]
df_train.Power[df_train.Power == 'null'] = np.NaN
df_train['Power'].isnull().sum()
df_train['Power'].fillna(value = 74, inplace = True)
df_train.isna().sum()
df_train['Name'] = df_train['Name'].str.split(' ').str[0]
df_train.groupby('Name')['id'].nunique()
df_train.Name[df_train.Name == 'Isuzu'] = 'ISUZU'
del df_train['New_Price']
dataset = df_train.copy()
del df_train['id']
df_train.dtypes
df_train['Year'] = df_train['Year'].astype(float)
df_train['Kilometers_Driven'] = df_train['Kilometers_Driven'].astype(float)
df_train['Price_log'] = np.log1p(df_train['Price'].values)
del df_train['Price']
df_train = pd.get_dummies(df_train, drop_first = True)
X = df_train.drop(columns = ['Price_log'], axis = 1)
y = df_train.iloc[:, 6].values
from sklearn.model_selection import train_test_split
import collections
from sklearn.metrics import r2_score
import re
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
%matplotlib inline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
regressor_1 = LinearRegression()
regressor_1.fit(X_train, y_train)
y_pred_1 = regressor_1.predict(X_test)
regressor_1.score(X_test,y_test)
regressor_2 = RandomForestRegressor(random_state = 0)
regressor_2.fit(X_train, y_train)
y_pred_2 = regressor_2.predict(X_test)
regressor_2.score(X_test,y_test)
regressor_3 = DecisionTreeRegressor(random_state = 0)
regressor_3.fit(X_train, y_train)
y_pred_3 = regressor_3.predict(X_test)
regressor_3.score(X_test, y_test)
plt.style.use('ggplot')
colors = ['#FF8C73','#66b3ff','#99ff99','#CA8BCA', '#FFB973', '#89DF38', '#8BA4CA', '#ffcc99', 
          '#72A047', '#3052AF', '#FFC4C4']
plt.figure(figsize = (10,8))
bar1 = sns.countplot(dataset['Year'])
bar1.set_xticklabels(bar1.get_xticklabels(), rotation = 90, ha = 'right')
plt.title('Count year wise', size = 24)
plt.xlabel('Year', size = 18)
plt.ylabel('Count', size = 18)
plt.show()
plt.figure(figsize = (5,5))
sns.countplot(dataset['Fuel_Type'])
plt.title('Types of Fuel and count', size = 24)
plt.tight_layout()
plt.show()
plt.figure(figsize = (5,5))
sns.countplot(dataset['Transmission'])
plt.title('Types of transmission', size = 24)
plt.tight_layout()
plt.show()


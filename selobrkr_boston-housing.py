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
import pandas as pd #for working with dataframe

import numpy as np #for linear algebra
import matplotlib.pyplot as plt #all for visualization

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/bostonhousing/BostonHousing.csv')
df.head()
df.info()
df.describe().transpose()
for i in df.columns: #first impression on variables 

    plt.figure(figsize=(8,6))

    sns.distplot(df[i].dropna())

    plt.show()
plt.figure(figsize=(12,6)) #correlation between variables

sns.heatmap(data=df.corr(), cmap='viridis', annot=True )
sns.jointplot(x='RM', y='MEDV', data=df, kind='scatter') #visualizing the higher correlation values between x-y
sns.jointplot(x='LSTAT', y='MEDV', data=df, kind='scatter') #visualizing the higher correlation values between x-y
sns.jointplot(x='AGE', y='NOX', data=df, kind='scatter') #visualizing the higher correlation values between variables
sns.jointplot(x='RAD', y='CRIM', data=df, kind='scatter') #visualizing the higher correlation values between variables
for i in df.columns: #boxplots for taking a glance at distribution of variables again

    sns.boxplot(df[i].dropna())

    plt.show()
df = df.drop(df.index[df['DIS'] > 10], axis=0) #getting rid of outliers
df = df.drop(df.index[df['PTRATIO'] < 13], axis=0) #getting rid of outliers
df = df.drop(df.index[df['LSTAT'] > 32], axis=0) #getting rid of outliers
df.columns
df['CRIM'] = df['CRIM'].fillna(value=df['CRIM'].median()) #handling the missing values
df['ZN'] = df['ZN'].fillna(value=df['ZN'].median()) #handling the missing values
df['RM'] = df['RM'].fillna(value=df['RM'].median()) #handling the missing values
df['AGE'] = df['AGE'].fillna(value=df['AGE'].mean()) #handling the missing values
df['INDUS'] = df['INDUS'].fillna(value=df['INDUS'].mean()) #handling the missing values
df['CHAS'].mode()
df['CHAS'] = df['CHAS'].fillna(value=0) #handling the missing values
sns.boxplot(x=df['LSTAT'], data=df)
df['LSTAT'] = df.drop(df.index[df['LSTAT'] > 31]) #getting rid of outliers
sns.boxplot(x=df['LSTAT'], data=df)
df['LSTAT'] = df['LSTAT'].fillna(value=df['LSTAT'].median()) #handling the missing values
df.isna().any() #is there anyone else?!
for i in df.columns:

    sns.boxplot(df[i])

    plt.show()
X = df.drop('MEDV', axis=1) #x and y values

y = df['MEDV']
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
from sklearn.preprocessing import RobustScaler #since we have too much outliers on variables
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(X_train, y_train)
pred = lm.predict(X_test)
pred
print(lm.intercept_)
lm.coef_
plt.scatter(y_test,pred) #not bad except for a couple of values

plt.xlabel('Y Test (True Values)')

plt.ylabel('Predicted Values')
sns.distplot((y_test-pred),bins=50) #seems normally distributed but we have outliers
from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, pred))

print('\n')

print('MSE: ', metrics.mean_squared_error(y_test, pred))

print('\n')

print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))
from sklearn.metrics import r2_score
metrics.explained_variance_score(y_test, pred) #not bad for a real life example
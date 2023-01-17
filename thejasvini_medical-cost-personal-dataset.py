# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
### Reading and Understanding the data
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
df= pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.head()
df.shape
df.info()
df.describe()
# Check for null count column wise

df.isnull().sum(axis=0)
plt.figure(figsize=(10,5))

sns.distplot(df['charges'])

plt.show()
plt.figure(figsize=(18,4))

plt.subplot(131)

sns.barplot(x='sex', y='charges', data=df)

plt.subplot(132)

sns.barplot(x='smoker', y='charges', data=df)

plt.subplot(133)

sns.barplot(x='region', y='charges', data=df)

plt.show()
sns.pairplot(df)
#Plot a heatmap and look at the corelation

sns.heatmap(df.corr(), cmap='coolwarm',annot=True)
# Let us map the variables with 2 levels to 0 and 1

df['sex']=df['sex'].map({'male':1, 'female':0})

df['smoker']=df['smoker'].map({'yes':1,'no':0})
# Assigning dummy variables to remaining categorical variable- region

df = pd.get_dummies(df, columns=['region'], drop_first=True)

df.head()
df_train, df_test= train_test_split(df, train_size=0.7, random_state=100)
df_train.shape
df_test.shape
## Rescaling features using min-max scaling

scaler = MinMaxScaler()

num_vars = ['age','bmi','children','charges']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.describe()
y_train = df_train.pop('charges')

X_train = df_train
y_train.head()
lm = LinearRegression()

lm.fit(X_train,y_train)
list(zip(X_train.columns,lm.coef_))
y_train_pred= lm.predict(X_train)

res= y_train- y_train_pred

sns.distplot(res)
r2_score(y_train, y_train_pred)
df_test[num_vars] = scaler.transform(df_test[num_vars])

df_test.describe()
y_test = df_test.pop('charges')

X_test = df_test
y_test_pred= lm.predict(X_test)

r2_score(y_test,y_test_pred)
# Plotting y_test and y_test_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_test_pred)

plt.xlabel('y_test', fontsize=18)                          

plt.ylabel('y_test_pred', fontsize=16) 
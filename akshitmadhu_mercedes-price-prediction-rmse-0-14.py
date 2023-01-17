# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import category_encoders as ce

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from scipy import stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/used-car-dataset-ford-and-mercedes/merc.csv')
df.head()
df.shape
df.dtypes
df.isnull().sum()
df.describe()
sns.countplot(df['transmission'])

plt.show()
print(df['model'].value_counts()/len(df))

plt.figure(figsize=(8,8))

sns.countplot(y = df['model'])

plt.show()
sns.countplot(df['fuelType'])

plt.show()
plt.figure(figsize=(15,5),facecolor='w')

sns.barplot(x= df['year'],y=df['price'])

plt.show()
sns.barplot(x= df['transmission'],y=df['price'])

plt.show()
sns.pairplot(df)

plt.show()
df_new = df.copy()
df_new.head()
df_new = pd.get_dummies(df)
df_new.head()
df_new.shape
sns.distplot(df_new.loc[:,'price'],norm_hist=True)

plt.title('Histogram Before Transformation of data')

plt.show()

print("Skewness: " + str(df_new['price'].skew()))

print("Kurtosis: " + str(df_new['price'].kurt()))
log_df = df_new.copy()
log_df['price'] = np.log1p(df_new['price'])
sns.distplot(log_df.loc[:,'price'],norm_hist=True)

plt.title('Histogram After transformation of data')

plt.show()

print("Skewness: " + str(log_df['price'].skew()))

print("Kurtosis: " + str(log_df['price'].kurt()))
log_df.head()
X = log_df.drop('price',axis=1)

y = log_df['price']
lr = LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)
lr.fit(x_train,y_train)
lr.score(x_test,y_test)
y_pred = lr.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("MSE:- {}".format(mse))

print("R2 Score:- {}".format(r2))

print("RMSE:- {}".format(rmse))
results = x_test.copy()

results['predicted'] = np.expm1(lr.predict(x_test))

results['actual'] = np.expm1(y_test)

results = results[['predicted','actual']]

results['predicted'] = results['predicted'].round(2)
results
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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/california-housing-prices/housing.csv')
df.head()
df.info()
df.describe()
df['ocean_proximity'].value_counts()
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False)

plt.title('Missing Data')

plt.show()
sns.boxplot(df['total_bedrooms'])
df['total_bedrooms']=df.total_bedrooms.fillna(df.total_bedrooms.median())
sns.heatmap(df.isnull(),yticklabels=False)

plt.title('Missing Data')

plt.show()
df.isnull().any()
sns.distplot(df['total_bedrooms'])
plt.figure(figsize=(6,8))

sns.distplot(df['median_house_value'],color='green',bins=50)

df[df['median_house_value']>450000]['median_house_value'].value_counts().head(10)
df=df.loc[df['median_house_value']<500001,:]
sns.distplot(df['median_house_value'],color='green')
sns.stripplot(df['population'],color='purple')
df=df.loc[df['population']<25000,:]
sns.pairplot(df)
plt.figure(figsize=(12,6))

sns.heatmap(data=df.corr(),annot=True)
df.head()
df.drop('ocean_proximity',axis=1,inplace=True)
df.head(3)
X=df.drop('median_house_value',axis=1)

y=df['median_house_value']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
print(lm.intercept_)
lm.score(X_train,y_train)
predictions=lm.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50)
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,predictions))

print('MSE:',metrics.mean_squared_error(y_test,predictions))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
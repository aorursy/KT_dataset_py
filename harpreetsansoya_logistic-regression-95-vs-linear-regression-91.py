import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df_train= pd.read_csv('../input/mobile-price-classification/train.csv')
df_test = pd.read_csv('../input/mobile-price-classification/test.csv')
df_train.head()
df_train.describe().T
df_train.columns

#df_test.columns
df_train.shape

#df_test.shape   (had same number of columns (21) and 1000 rows)
df_train.info()

#df_test.info()
df_train.isnull().sum()

#df_test.isnull().sum()
df_train.duplicated(keep=False).any()
labels= ['Supported 3G', ' Not-Supported']

values=df_train['three_g'].value_counts().values
plt.axis('equal')

plt.pie(values, labels=labels,autopct='%0.00f%%', explode=[0.1,0], shadow=True)

plt.show
labels= ['Supported 4G', ' Not-Supported']

values=df_train['four_g'].value_counts().values
plt.axis('equal')

plt.pie(values, labels=labels,autopct='%0.00f%%', explode=[0.1,0], shadow=True)

plt.show
sns.countplot(df_train['dual_sim'])
sns.distplot(df_train['fc'], color='red', kde=False)
sns.distplot(df_train['battery_power'],color='maroon', kde=False)
sns.distplot(df_train['mobile_wt'],color='teal', kde=False)
sns.distplot(df_train['sc_w'],color='chocolate', kde=False)
plt.figure(figsize=(12, 9))

corr=df_train.corr()

sns.heatmap(corr[(corr<=0.5) | (corr>=-0.5)],vmin=-1, vmax=1, annot=True)
sns.boxplot(df_train['dual_sim'],df_train['price_range'])
sns.boxplot(df_train['four_g'],df_train['price_range'])
sns.boxplot(df_train['three_g'],df_train['price_range'])
sns.jointplot(x='price_range', y='ram', data=df_train, kind='kde')
sns.boxplot(df_train['dual_sim'],df_train['price_range'])
sns.boxplot(df_train['wifi'],df_train['price_range'])
sns.boxplot(df_train['touch_screen'],df_train['price_range'])
#Since all the features are in different range, preprocessing scalar is applied

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



scaler = StandardScaler()

X = df_train.drop('price_range',axis=1)

y = df_train['price_range']



scaler.fit(X)

X_transformed = scaler.transform(X)



X_train,X_test,y_train,y_test = train_test_split(X_transformed,y,test_size=0.3, random_state=31)
from sklearn.linear_model import LogisticRegression

model= LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
from sklearn.metrics import confusion_matrix
y_pred= model.predict(X_test)
cm= confusion_matrix(y_test,y_pred)
plt.figure(figsize= (8,5))

sns.heatmap(cm, annot=True)

plt.xlabel= 'predict'

plt.ylabel='truth'
from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(X_train,y_train)
lr.score(X_test,y_test)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
df.describe()
df.info()
sns.countplot(x='gender',data=df)
sns.countplot(x='race/ethnicity',data=df)
fig = plt.figure(figsize=(12,6))

sns.countplot(x='parental level of education',data=df)

fig.show()
sns.countplot(x='lunch',data=df)
sns.countplot('test preparation course',data=df)
sns.distplot(df['math score'])
sns.distplot(df['reading score'])
sns.distplot(df['writing score'])
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
#Student performing good on one subject is expected to score good in the remaining too subjects
sns.scatterplot(x='math score',y='reading score',data=df)
sns.scatterplot(x='reading score',y='writing score',data=df)
sns.scatterplot(x='math score',y='writing score',data=df)
df.groupby(by='gender').mean()
df.groupby(by='gender').mean().plot.bar()
df.groupby(by='race/ethnicity').mean()
df.groupby(by=['race/ethnicity']).mean().plot.bar()
df.groupby(by='parental level of education').mean()
df.groupby(by=['parental level of education']).mean().plot.bar()
df.groupby(by='lunch').mean()
df.groupby(by=['lunch']).mean().plot.bar()
df.groupby(by=['test preparation course']).mean()
df.groupby(by=['test preparation course']).mean().plot.bar()
#End of Explorary Data Analysis
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error
df1 = pd.get_dummies(data=df,drop_first=True)
df1.head()
X = df1.drop(['math score','reading score','writing score'],axis=1)

y_maths = df1['math score']

y_reading = df1['reading score']

y_writing = df1['writing score']
X_train, X_test, y_train_maths, y_test_maths = train_test_split(X, y_maths, test_size=0.33, random_state=42)

X_train, X_test, y_train_reading, y_test_reading = train_test_split(X, y_reading, test_size=0.33, random_state=42)

X_train, X_test, y_train_writing, y_test_writing = train_test_split(X, y_writing, test_size=0.33, random_state=42)
rfr = RandomForestRegressor(min_samples_leaf=20)
rfr.fit(X_train,y_train_maths)

pred_maths = rfr.predict(X_test)

print(mean_absolute_error(y_test_maths,pred_maths))
rfr.fit(X_train,y_train_reading)

pred_reading = rfr.predict(X_test)

print(mean_absolute_error(y_test_reading,pred_reading))
rfr.fit(X_train,y_train_writing)

pred_writing = rfr.predict(X_test)

print(mean_absolute_error(y_test_writing,pred_writing))
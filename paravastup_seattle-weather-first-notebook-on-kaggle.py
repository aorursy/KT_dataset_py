import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from datetime import date
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/seattleWeather_1948-2017.csv')
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Converting the date to datetime object

df['DATE'] = pd.to_datetime(df['DATE'], errors = 'coerce')
df['DAY'] = df['DATE'].dt.weekday_name

df['DAY_OF_WEEK'] = df['DATE'].apply(date.isoweekday)

df['MONTH'] = df['DATE'].dt.month

df['YEAR'] = df['DATE'].dt.year

df['RAIN'] = df['RAIN'].apply(lambda x: 1 if x == True else 0)
df.head()
plt.figure(figsize=(12,5))

ax = sns.countplot(data = df[(df['RAIN'] == True) & (df['YEAR'] >= 2015)], x='MONTH', hue='YEAR')
int(df[(df['RAIN'] == True) & (df['YEAR'] >= 2000)].groupby('YEAR').count()['DAY'].mean())
plt.figure(figsize=(12,5))

ax = sns.countplot(data = df[(df['RAIN'] == True) & (df['YEAR'] >= 2000)], x='YEAR')

plt.tight_layout()
g = sns.lmplot(x='TMIN', y= 'PRCP', data=df, fit_reg=False, hue='RAIN', size=8)
g = sns.lmplot(x='TMAX', y= 'PRCP', data=df, fit_reg=False, hue='RAIN', size=8)
g = sns.lmplot(x='TMIN', y= 'TMAX', data=df, fit_reg=False, hue='RAIN', size=8)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(df.drop(['DATE', 'DAY', 'YEAR', 'DAY_OF_WEEK', 'RAIN'],axis=1), 

                                                    df['RAIN'], test_size=0.30)
y_train = y_train.fillna(y_train.mean())

X_train = X_train.fillna(X_train.mean())

X_test = X_test.fillna(X_test.mean())
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
logpredictions = logmodel.predict(X_test)
print(accuracy_score(y_test,logpredictions))
print(confusion_matrix(y_test,logpredictions))
print(classification_report(y_test,logpredictions))
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree_predictions = dtree.predict(X_test)
print(accuracy_score(y_test,dtree_predictions))
print(classification_report(y_test,dtree_predictions))
print(confusion_matrix(y_test,dtree_predictions))
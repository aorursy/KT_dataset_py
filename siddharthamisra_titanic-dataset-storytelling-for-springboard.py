#Import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Load training data
df_train = pd.read_csv('../input/titanic/train.csv')
df_test = pd.read_csv('../input/titanic/train.csv')
df = df_train.append(df_test)
df.head()
df.describe()
df.hist(figsize=(20,6))
df.dtypes
df.Fare.sort_values(ascending=False)
df.Fare.quantile(.97)
#Drop descriptive features
df = df.drop(['Name', 'Ticket','Cabin','PassengerId'], axis=1)
#Impute missing Age values with mean
df.Age = df.Age.fillna(df.Age.mean())
#Perform listwise deletion of top 3% of fares paid
df = df[df.Fare < df.Fare.quantile(.97)]
#Assign categorical variables
df.Sex = df.Sex.astype('category')
df.Embarked = df.Embarked.astype('category')
df.describe()
df.hist(figsize=(20,6))
df.Parch = df.Parch.apply(lambda x: 1 if x > 0 else 0)
df.SibSp = df.SibSp.apply(lambda x: 1 if x > 0 else 0)
df = pd.get_dummies(df)
df.corr()
X = df.drop('Survived', axis=1)
y = df.Survived
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train_scaled, y_train)
clf.score(X_test_scaled,y_test)
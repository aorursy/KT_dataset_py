import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
df = pd.read_csv('../input/titanic/train.csv')
df1 = pd.read_csv('../input/titanic/test.csv')
df.head()
df1.head()
sns.barplot(x='Sex',y='Survived',data=df)
sns.barplot(x='Pclass',y='Survived',data=df)
sns.barplot(x='Embarked',y='Survived',data=df)
sns.barplot(x='SibSp',y='Survived',data=df)
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)
df.isnull().sum()
df1.isnull().sum()
df.drop(columns='PassengerId',inplace=True)
df.drop(columns='Cabin',inplace=True)
df.drop(columns='Ticket',inplace=True)
df.drop(columns='Name',inplace=True)
df.drop(columns='Fare',inplace=True)
df1.drop(columns='Name',inplace=True)
df1.drop(columns='Ticket',inplace=True)
df1.drop(columns='Cabin',inplace=True)
df1.drop(columns='PassengerId',inplace=True)
df1.drop(columns='Fare',inplace=True)
df.head()
df1.head()
df.info()
df.describe()
df.head()
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
df1['Age'].fillna(df1['Age'].mean(),inplace=True)
df.head()
df1.head()
df.isnull().sum()
df1.isnull().sum()
X = df.iloc[:,1:].values
y = df.iloc[:, 0].values
X_test = df1.iloc[:,:].values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test))
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.tree import DecisionTreeClassifier
d = DecisionTreeClassifier(max_depth = 5)
d.fit(X,y)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X, y)
y_kpred = classifier.predict(X_test)
y_kpred
y_test = d.predict(X_test)
y_test
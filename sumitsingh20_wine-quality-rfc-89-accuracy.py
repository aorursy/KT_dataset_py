import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.info()
df.isna().sum()
df.head()
df['quality'].value_counts()
sns.countplot(df['quality']);
sns.barplot(x = df['quality'],y = df['fixed acidity']);
sns.barplot(x = df['quality'],y = df['volatile acidity']);
sns.boxplot(x = df['quality'],y = df['citric acid']);
sns.scatterplot(x = df['quality'],y = df['residual sugar']);
sns.boxplot(x = df['quality'],y = df['chlorides']);
sns.boxplot(x = df['quality'],y = df['free sulfur dioxide']);
sns.boxplot(x = df['quality'],y = df['total sulfur dioxide']);
sns.scatterplot(x = df['quality'],y = df['density']);
sns.scatterplot(x = df['quality'],y = df['pH']);
sns.barplot(x = df['quality'],y = df['sulphates']);
sns.boxplot(x = df['quality'],y = df['alcohol']);
corr = df.corr()

sns.heatmap(corr)
df.describe()
df.head()
bins = (2, 6.5, 8)

group_names = ['0', '1']

df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
df['quality'].value_counts()
sns.countplot(df['quality'])
x = df.drop('quality',axis = 1)

y = df['quality']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
model = RandomForestClassifier(n_estimators = 200,random_state = 1)

model.fit(x_train,y_train)
model.score(x_test,y_test)
y_preds = model.predict(x_test)
print(classification_report(y_test,y_preds))
print(confusion_matrix(y_test,y_preds))
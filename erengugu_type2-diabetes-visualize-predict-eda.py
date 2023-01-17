# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/type2-diabetes/diabetes.csv")
df.head()
df.info()
df.shape
df.isnull().sum()
df.describe().T
sns.heatmap(df.isnull());
sns.distplot(df.Pregnancies);
sns.distplot(df.Glucose,kde=False);
sns.jointplot(x='Pregnancies', y='Glucose', data=df, kind='reg');
sns.jointplot(x='Age', y='Glucose', data=df, kind='reg');
sns.kdeplot(df["Age"])
sns.kdeplot(df["Pregnancies"]);
sns.pairplot(df);
sns.pairplot(df,hue="Outcome");
plt.figure(figsize=(18,8))

sns.barplot(x="Age",y="Outcome",data=df);
sns.barplot(x="Pregnancies",y="Outcome",data=df);
df.Pregnancies.value_counts().sort_values(ascending=False)
sns.countplot(x="Pregnancies",data=df);
df.Outcome.value_counts().sort_values(ascending=False)
sns.countplot(x="Outcome",data=df);
df.dtypes
corr = df.corr()

plt.figure(figsize=(14,12))

sns.heatmap(corr,annot=True,square=True,cmap="Blues");
df.Glucose.sort_values(ascending=False).head()
sns.jointplot(x='Glucose', y='Outcome', data=df, kind='reg');

sns.kdeplot(df['Glucose'])

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']]

y = df["Outcome"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)

print(accuracy_score(y_test,y_pred))
tree = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)

print(accuracy_score(y_test,y_pred))
tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=0)

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)

print(accuracy_score(y_test,y_pred))
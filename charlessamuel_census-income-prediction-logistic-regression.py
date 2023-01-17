# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
df = pd.read_csv('../input/adult-census-income/adult.csv', na_values=['?'])

df.sample(10)
df.income.value_counts()
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

df.head()
df.income.value_counts()
df.shape
df.dtypes
df.info()
df.isnull().sum().any()
df.isnull().sum()
print("Types of working class :\n", df.workclass.value_counts(), '\n', 'Types in occupation\n', df.occupation.value_counts(), '\n', 'Types in Native Country\n', df['native.country'].value_counts())
df['workclass'] = df['workclass'].fillna('X')

df['occupation'] = df['occupation'].fillna('X')

df['native.country'] = df['native.country'].fillna('X')
df.dtypes
num_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'income']



cat_features = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
#Income counts

sns.countplot(df['income'])

plt.show()
#Plotting for the numerical features

fig, ax = plt.subplots(figsize=(25, 25))

p = sns.heatmap(df[num_features].corr(), annot=True, cmap="Blues")

plt.title("Correlation of Numerical Features", fontsize=20)

plt.show()
#Education Number vs Income

g = sns.catplot(x="education.num",y="income",data=df,kind="bar",height = 6,palette = "muted")

g.despine(left=True)

g = g.set_ylabels(">50K probability")
df['sex'].value_counts()
df['marital.status'].value_counts()
#Map Sex as a binary column

df['sex'] = df.sex.map({'Male':0, 'Female':1})



#Married can be converted manually to binary columns

#No spouse means Single

df['marital.status'] = df['marital.status'].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'], 'Single')

#Spouse means Married More simple right

df['marital.status'] = df['marital.status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')

#Now we map it to Binary values

df['marital.status'] = df['marital.status'].map({'Married':1, 'Single':0})
#Drop columns that do not contribute to final result

df.drop(labels=["workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)

df.head()
X = df.drop(labels=['income'], axis=1)

y = df['income']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
model = LogisticRegression()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("Accuracy: %s%%" % (100*accuracy_score(y_test, y_pred)))

print(confusion_matrix(y_test, y_pred))

print("Classification Report for Logistic Regression")

print(classification_report(y_test, y_pred))
import numpy as np

import pandas as pd

from pandas import Series,DataFrame



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv',index_col='PassengerId')

df.head()
df.info()
sns.countplot(y='Country',data=df)
sns.countplot('Sex',data=df)
sns.kdeplot(df['Age'])

x_min = df['Age'].min()

x_max = df['Age'].max()

plt.xlim(x_min,x_max)
fig = sns.FacetGrid(df,hue='Sex',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

x_min = df['Age'].min()

x_max = df['Age'].max()

fig.set(xlim=(x_min,x_max))
sns.countplot('Category',data=df)
fig2 = sns.FacetGrid(df,hue='Category',aspect=4)

fig2.map(sns.kdeplot,'Age')

fig2.set(xlim=(0,x_max))
sns.countplot('Survived',data=df)
ax1 = sns.FacetGrid(df,hue='Survived',aspect=4)

ax1.map(sns.kdeplot,'Age')

ax1.set(xlim=(0,x_max))

ax1.add_legend()
sns.countplot('Sex',hue='Survived',data=df)
sns.countplot('Category',hue='Survived',data=df)
s_rate = df.groupby(['Survived','Category'])['Category'].count()

print(s_rate)



cat = df.groupby('Category')['Category'].count()

print(cat)



x = (s_rate[1]['C'])/cat['C']

y = (s_rate[1]['P'])/cat['P']

print(f'Survival rate of crew is {x}')

print(f'Survival rate of passengers is {y}')
df_copy = df.copy()
df_copy = df_copy.drop(['Firstname','Lastname'],axis=1)
from sklearn.preprocessing import LabelEncoder
def encoder (value):

    encode = LabelEncoder().fit(value)

    return encode.transform(value)
df_copy['Sex'] = encoder(df_copy['Sex'])

df_copy['Category'] = encoder(df_copy['Category'])

df_copy['Country'] = encoder(df_copy['Country'])

df_copy.head()
"""Normalising to a range of 0=10"""

def normalize(values):

    mn = values.min()

    mx = values.max()

    return(10.0/(mx - mn) * (values - mx)+10)
df_copy = normalize(df_copy)
df_copy.describe()
sns.heatmap(df_copy.corr(),annot=True)
Y = df_copy['Survived']

X = df_copy.drop(['Survived'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y)
from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
lreg.fit(x_train,y_train)
y_pred = lreg.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
from sklearn.model_selection import cross_val_predict
lreg2 = LogisticRegression()

cross_y_pred = cross_val_predict(lreg2,X,Y,cv=5)
accuracy_score(Y,cross_y_pred)
x_train,x_test,y_train,y_test = train_test_split(X,Y)
from sklearn.svm import SVC
clf = SVC(random_state=1)
clf.fit(x_train,y_train)
y_pred1 = clf.predict(x_test)
accuracy_score(y_test,y_pred1)
clf2 = SVC(random_state=1)

cross_y_pred1 = cross_val_predict(clf2,X,Y,cv=5)
accuracy_score(Y,cross_y_pred1)
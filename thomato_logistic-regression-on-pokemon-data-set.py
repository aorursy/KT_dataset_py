import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os



print(os.listdir('../input'))

%matplotlib inline
df = pd.read_csv('../input/pokemon_alopez247.csv')
df.head()
df.info()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

df.info()
leg = df[df['isLegendary'] == True]

sns.countplot(x='hasGender', data=leg)
leg_total = leg['Total'].mean()

non_leg_total = df[df['isLegendary'] != True]['Total'].mean()
pd.DataFrame([leg_total, non_leg_total], index=['Legendary', 'non-Legendary'], columns=['Average Total'])
plt.figure(figsize=(15,6))

plt.title("Catch rate of Poekemons")

sns.scatterplot(x='Number', y='Catch_Rate', data=df, hue='isLegendary')
isLegendary = pd.get_dummies(df['isLegendary'], drop_first=True)

hasGender = pd.get_dummies(df['hasGender'], drop_first=True)

lr_df = df[['Total', 'Catch_Rate']]
lr_df = pd.concat([lr_df, isLegendary, hasGender], axis=1)
lr_df.columns = ['Total', 'Catch_Rate', 'isLegendary', 'hasGender']
lr_df.head()
X = lr_df.drop('isLegendary', axis=1)

y = lr_df['isLegendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) 
LR = LogisticRegression()

LR.fit(X_train, y_train)
pred = LR.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
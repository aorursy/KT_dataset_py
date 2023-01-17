import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
print(df.head())

print('\n')

print(df.info())

print('\n')

print(df.describe())
plt.figure(figsize=(12,8))

s = sns.countplot(x='class',data=df)

for p in s.patches:

    s.annotate(format(p.get_height(),'.1f'),

              (p.get_x() + p.get_width() /2. , p.get_height()),

               ha='center',va='center',fontsize=15,xytext=(0,9), textcoords='offset points'

              )
columns = df.columns

print(columns)
f, axes = plt.subplots(22,1, figsize=(15,150), sharey = True)

k=0

for i in range(0,22):

    k = k+1

    s = sns.countplot(x=columns[k],data=df,ax=axes[i])

    for p in s.patches:

        s.annotate(format(p.get_height(), '.1f'), 

        (p.get_x() + p.get_width() / 2., p.get_height()), 

        ha = 'center', va = 'center', 

        xytext = (0, 9), 

        fontsize = 15,

        textcoords = 'offset points')

  
df = df.drop('veil-type',axis=1)
#Reassigning because one feature was dropped from the dataframe

columns = df.columns
f, axes = plt.subplots(21,1, figsize=(15,150), sharey = True)

k=0

for i in range(0,21):

    k = k+1

    s = sns.countplot(x=columns[k],data=df,ax=axes[i],hue='class')

    for p in s.patches:

        s.annotate(format(p.get_height(), '.1f'), 

        (p.get_x() + p.get_width() / 2., p.get_height()), 

        ha = 'center', va = 'center', 

        xytext = (0, 9), 

        fontsize = 15,

        textcoords = 'offset points')

  
#Checking for null values

sns.heatmap(df.isnull())
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
#Getting data ready to use for model training

final_df = pd.get_dummies(df,drop_first=True)
print(final_df.head())

print('\n')

print(final_df.info())

print('\n')
X = final_df.drop('class_p',axis=1) #Everything except target

y = final_df['class_p'] #Only target



#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
pred_rfc = rfc.predict(X_test)
print('CONFUSION MATRIX')

print(confusion_matrix(y_test,pred_rfc))

print('\n')

print('CLASSIFICATION REPORT')

print(classification_report(y_test,pred_rfc))

print('\n')

print('ACCURACY')

print(accuracy_score(y_test,pred_rfc))
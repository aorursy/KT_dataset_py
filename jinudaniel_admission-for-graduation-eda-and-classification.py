import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
df.head()
df.columns
df.drop(['Serial No.'], inplace=True, axis=1)
df.head()
df.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'}, inplace=True)
df.columns
plt.figure(figsize=(15, 6))
sns.barplot(x='GRE Score', y='Chance of Admit', data=df)
plt.figure(figsize=(15, 6))
sns.barplot(x='TOEFL Score', y='Chance of Admit', data=df)
plt.figure(figsize=(15, 9))
sns.boxplot(x='CGPA', y='Chance of Admit', data=df)
plt.figure(figsize=(15, 6))
sns.barplot(x='SOP', y='Chance of Admit', data=df)
plt.figure(figsize=(15, 6))
sns.barplot(x='LOR', y='Chance of Admit', data=df)
plt.figure(figsize=(15, 9))
sns.scatterplot(x='GRE Score', y='TOEFL Score', hue='Research', data=df)
def hasAdmitted(data):
    if data > 0.7:
        return 1
    else:
        return 0
df['Admit'] = df['Chance of Admit'].apply(hasAdmitted)
df.head()
df.drop(['Chance of Admit'], inplace=True, axis=1)
df.head()
print(df['Admit'].value_counts())
plt.figure(figsize=(15, 9))
sns.countplot(x='Admit', data=df)
#plt.figure(figsize=(15, 9))
sns.scatterplot(x='GRE Score', y='TOEFL Score', hue='Admit', data=df)
#plt.figure(figsize=(15, 9))
sns.scatterplot(x='SOP', y='LOR', hue='Admit', data=df)
sns.scatterplot(x='GRE Score', y='CGPA', hue='Admit', data=df)
sns.scatterplot(x='TOEFL Score', y='CGPA', hue='Admit', data=df)
sns.heatmap(df.corr(), annot=True)
X_train,X_test,y_train,y_test=train_test_split(df.drop(['Admit'], axis=1),df['Admit'],
                                               test_size=0.1,random_state=50)
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test,y_test)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
feat_importances = pd.Series(rf.feature_importances_, index=df.drop('Admit', axis=1).columns)
feat_importances.sort_values(ascending=False).plot(kind='barh')

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.DataFrame(pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv'))
df.head()
plt.figure(figsize=(10,7))
sns.distplot(df['Time'])
plt.figure(figsize=(10,7))
sns.distplot(df['Amount'])
sns.countplot(x='Class',data=df)
print("Percentage of non-fraud transactions:", round(100*df[df['Class']==0]['Class'].value_counts()/len(df),2))
print("Percentage of fraud transactions:", round(100*df[df['Class']==1]['Class'].value_counts()/len(df),2))
sns.stripplot(x='Class', y='Amount',data=df)
df.describe()
plt.figure(figsize=(30, 25))
sns.heatmap(df.corr(), annot=True)
sns.scatterplot(x='Amount',y='V2',data=df)
sns.scatterplot(x='Amount', y='V5',data=df)
sns.scatterplot(x='Amount', y='V7',data=df)
sns.scatterplot(x='Amount', y='V20',data=df)
plt.figure(figsize=(10,7))
sns.scatterplot(x='Time', y='V3',data=df)
from sklearn.preprocessing import RobustScaler
robsc = RobustScaler()
df['scaled_amount'] = robsc.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = robsc.fit_transform(df['Time'].values.reshape(-1,1))
df.drop('Amount', inplace=True, axis=1)
df.drop('Time', inplace=True, axis=1)
display(df.head())
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount','scaled_time'], axis=1, inplace=True)

df.insert(0,'scaled_amount', scaled_amount)
df.insert(1,'scaled_time', scaled_time)
from sklearn.model_selection import train_test_split

X=df.drop('Class', axis=1)
y=df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=df['Class'])
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "Support Vector Classifier": SVC()
}
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import TomekLinks

tl = TomekLinks()
X_tomek, y_tomek= tl.fit_sample(X_train, y_train)
from collections import Counter
print('Resampled dataset shape %s' % Counter(y_tomek))
for key, classifier in classifiers.items():
    classifier.fit(X_tomek, y_tomek)
    print("Classifiers: ", classifier.__class__.__name__)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
from imblearn.over_sampling import SMOTE

sm = SMOTE()
X_sm, y_sm= sm.fit_sample(X_train, y_train)
for key, classifier in classifiers.items():
    classifier.fit(X_sm, y_sm)
    print("Classifiers: ", classifier.__class__.__name__)
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
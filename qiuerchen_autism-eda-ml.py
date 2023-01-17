# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/autism-screening-for-toddlers/Toddler Autism dataset July 2018.csv')
df.columns

df.drop(['Case_No','Who completed the test'], axis=1, inplace=True)

df.columns
import seaborn as sns
sns.countplot(df['Class/ASD Traits '])
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(df.corr(),annot=True,cmap='coolwarm',vmin=0, vmax=1)
sns.scatterplot(x='Age_Mons',y='Qchat-10-Score',data=df)
sns.scatterplot(x='A10',y='Qchat-10-Score',data=df)
sns.countplot(df['Ethnicity'])
plt.xticks(rotation=90)
# df.columns
# df_new = df[df['Class/ASD Traits ']=='Yes']
sns.countplot(x='Ethnicity',hue='Sex',data=df)
plt.xticks(rotation=90)

#we can see that more survey respondants are males than females for each ethnicity
df_plot = df.groupby(['Sex', 'Ethnicity']).size().reset_index().pivot(columns='Sex', index='Ethnicity', values=0)
# df_plot
df_plot.plot(kind='bar', stacked=True)
x, y, hue = 'Ethnicity', 'prop', 'Sex'
prop_df = (df[x]
            .groupby(df[hue])
            .value_counts(normalize=True)
            .rename(y)
            .reset_index())
sns.barplot(x=x,y=y,hue=hue,data=prop_df)
plt.xticks(rotation=90)
# sns.countplot(df['Sex'])
df = df[df['Class/ASD Traits ']=='Yes']
sns.countplot(df['Sex'])
#we can see more males are are categorized as ASD thru the app
df_yes = df[df['Class/ASD Traits ']=='Yes']

m_prop = df_yes.Sex.value_counts()['m']/df.Sex.value_counts()['m']
f_prop = df_yes.Sex.value_counts()['f']/df.Sex.value_counts()['f']
m_prop
f_prop
df.info()
from sklearn.preprocessing import OneHotEncoder
cat_features = df.select_dtypes(include='object')
num_features = df.select_dtypes(exclude='object')
df = pd.get_dummies(df, drop_first=True)
df
from sklearn.model_selection import train_test_split
X = df.drop(['Class/ASD Traits _Yes'], axis=1)
y = df['Class/ASD Traits _Yes']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#logistic method
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
preds = logreg.predict(X_test)
logreg.score(X_train, y_train)
#classification report
#confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
classification_report(y_test, preds)

confusion_matrix(y_test,preds)
#Let's try another model
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
classification_report(y_test, pred)

confusion_matrix(y_test, pred)
#to get the optimum n_neighbors
#this code is taken from a Udemy course "Python for Data Science and Machine Learning" taught by Jose Portilla
import matplotlib.pyplot as plt
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    error_rate.append(np.mean(y_test != pred))
plt.figure(figsize=(10,10))
plt.plot(range(1,40), error_rate, color='blue',linestyle='dashed',marker='o',markerfacecolor='red')
knn = KNeighborsClassifier(n_neighbors=27)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
classification_report(y_test, pred)
confusion_matrix(y_test, pred)
#decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
classification_report(y_test,pred)
confusion_matrix(y_test,pred)
#let's try random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
classification_report(y_test,pred)
confusion_matrix(y_test,pred)
#one last model, which I will use SVM
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)
classification_report(y_test,pred)
confusion_matrix(y_test,pred)
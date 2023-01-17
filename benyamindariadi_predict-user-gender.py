import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 
df=pd.read_csv('/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv',encoding = "ISO-8859-1")
df.info()

df
df.corr()
df.gender.unique()
df = df.dropna(subset=['gender'])



df.gender.unique()

df = df[df.gender != 'unknown']

df = df[df.gender != 'brand']
df=df[(df['gender:confidence'] >= 0.8)&(df['profile_yn:confidence'] >= 0.8)]
df.info()

df.isnull().sum()
#drop all columns with so many of unique value to simpify the dataset

df=df.drop('gender_gold',axis=1)

df=df.drop('profile_yn_gold',axis=1)

df=df.drop('tweet_coord',axis=1)

df=df.drop('tweet_location',axis=1)

df=df.drop('user_timezone',axis=1)

df=df.drop('_unit_id',axis=1)

df=df.drop('_last_judgment_at',axis=1)

df=df.drop('created',axis=1)

df=df.drop('name',axis=1)

df=df.drop('profileimage',axis=1)

df=df.drop('tweet_created',axis=1)

df=df.drop('_trusted_judgments',axis=1)

df=df.drop('gender:confidence',axis=1)

df=df.drop('profile_yn:confidence',axis=1)

df=df.drop('_golden',axis=1)

df=df.drop('text',axis=1)

df=df.drop('description',axis=1)

df=df.drop('link_color',axis=1)

df.isnull().sum()
df.nunique().sum
df['sidebar_color'] = df.sidebar_color.str[0] 
dummies= pd.get_dummies(df[['_unit_state','profile_yn','sidebar_color','tweet_id']],drop_first=True)

df=pd.concat([df.drop(['_unit_state','profile_yn','sidebar_color','tweet_id'], axis=1), dummies],axis=1) 
df.gender = df.gender.replace({'male': 1, 'female': 0})
df.corr()

plt.figure(figsize=(12,7))

sns.heatmap(df.corr(),annot=True,cmap='viridis')

sns.distplot(df['gender'],kde=False,bins=40)
df.corr()['gender'].sort_values().plot(kind='bar')
sns.countplot(x='gender',data=df)
x = df.drop('gender',axis=1).values

y = df['gender'].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
scores=[]

for i in range(1,30):

  tree=DecisionTreeClassifier(max_depth = i) 

  tree.fit(x_train, y_train) 

  scores.append(tree.score(x_test,y_test)) 

plt.plot(range(1,30),scores) 

plt.show()
tree=DecisionTreeClassifier(max_depth=5) 

tree.fit(x_train, y_train) 

tree.score(x_test,y_test)
predictions = tree.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))
from sklearn.neighbors import KNeighborsClassifier



accuracies=[]

for k in range(1,101):

  classifier = KNeighborsClassifier(n_neighbors = k)

  classifier.fit(x_train, y_train)

  accuracies.append(classifier.score(x_test, y_test)) 

  

k_list=list(range(1,101)) 

plt.plot(k_list,accuracies)

plt.show() 
classifier = KNeighborsClassifier(n_neighbors =29)

classifier.fit(x_train, y_train)

classifier.score(x_test, y_test) 
predictions = classifier.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))

from sklearn.svm import SVC



scores=[]

for i in (np.linspace(0.01,1,10)):

  classifier = SVC(kernel = 'linear', C = i)

  classifier.fit(x_train,y_train) 

  scores.append(classifier.score(x_test,y_test)) 

plt.plot(np.linspace(0.01,1,10),scores) 

plt.show()

scores=[]

for i in (np.linspace(0.01,1,10)):

  classifier = SVC(kernel = 'rbf', gamma=i, C = i)

  classifier.fit(x_train,y_train) 

  scores.append(classifier.score(x_test,y_test)) 

plt.plot(np.linspace(0.01,1,10),scores) 

plt.show()
classifier = SVC(kernel = 'rbf', gamma=0.8, C = 0.8)

classifier.fit(x_train,y_train) 

classifier.score(x_test,y_test) 
predictions = classifier.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))
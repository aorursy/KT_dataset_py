import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 
df=pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
df.info()

df

df.isnull().sum()

df.nunique().sum
df.department=df.department.isna()

df.company_profile=df.company_profile.isna()

df.requirements=df.requirements.isna()

df.benefits=df.benefits.isna()

df.industry=df.industry.isna()

df.function=df.function.isna()

df[['department','company_profile','requirements','benefits','industry','function']] = df[['department','company_profile','requirements','benefits','industry','function']].replace({True: 1, False: 0})
df = df.dropna(subset=['location','description'])
df.isnull().sum()
df['location'] = df['location'].str.split(',').str[0]
df['salary_range'] = df['salary_range'].str.split('-').str[-1]

df['salary_range'] = df['salary_range'].fillna(0)

df = df[df.salary_range != 'Nov']

df = df[df.salary_range != 'Oct']

df = df[df.salary_range != 'Dec']

df = df[df.salary_range != 'Apr']

df = df[df.salary_range != 'Jun']

df = df[df.salary_range != 'Sep']
df[['employment_type','required_experience','required_education']] = df[['employment_type','required_experience','required_education']].fillna('Other')
df=df.drop('title',axis=1)

df=df.drop('description',axis=1)

df=df.drop('job_id',axis=1)
dummies= pd.get_dummies(df[['location','employment_type','required_experience','required_education']],drop_first=True)

df=pd.concat([df.drop(['location','employment_type','required_experience','required_education'], axis=1), dummies],axis=1) 
df.isnull().sum()
df.nunique().sum
df
df.info

sns.countplot(x='fraudulent',data=df)

#The data is very skew
x = df.drop('fraudulent',axis=1).values

y = df['fraudulent'].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
scores=[]

for i in range(1,100):

  tree=DecisionTreeClassifier(max_depth = i) 

  tree.fit(x_train, y_train) 

  scores.append(tree.score(x_test,y_test)) 

plt.plot(range(1,100),scores) 

plt.show()

tree=DecisionTreeClassifier(max_depth =18) 

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
classifier = KNeighborsClassifier(n_neighbors =10)

classifier.fit(x_train, y_train)

classifier.score(x_test, y_test) 
predictions = classifier.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))

from sklearn.svm import SVC

scores=[]

for i in (np.arange(0.01,1,0.02)):

  classifier = SVC(kernel = 'linear', C = i)

  classifier.fit(x_train,y_train) 

  scores.append(classifier.score(x_test,y_test)) 

plt.plot(np.arange(0.01,1,0.02),scores) 

plt.show()
scores=[]

for i in (np.arange(0.01,1,0.02)):

  classifier = SVC(kernel = 'rbf', gamma=i, C = i)

  classifier.fit(x_train,y_train) 

  scores.append(classifier.score(x_test,y_test)) 

plt.plot(np.arange(0.01,1,0.02),scores) 

plt.show()
classifier = SVC(kernel = 'rbf', gamma=0.96, C = 0.96)

classifier.fit(x_train,y_train) 

classifier.score(x_test,y_test)
predictions = classifier.predict(x_test) 

from sklearn.metrics import classification_report,confusion_matrix 

print(classification_report(y_test,predictions)) 

print(confusion_matrix(y_test,predictions))
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

model = Sequential()

model.add(Dense(units=100,activation='relu'))

model.add(Dropout(0.5))





model.add(Dense(units=1,activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])



#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)



model.fit(x_train,y_train,epochs=65,validation_data=(x_test, y_test), verbose=1)

losses = pd.DataFrame(model.history.history)



losses[['accuracy','val_accuracy']].plot()

losses[['loss','val_loss']].plot()

print(model.metrics_names) 

print(model.evaluate(x_test,y_test,verbose=0))
from sklearn.metrics import classification_report,confusion_matrix



predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))
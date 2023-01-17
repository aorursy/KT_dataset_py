

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.shape
df.head()
df.tail()
df.info()
df.describe()
plt.figure(figsize=(15,6))

sns.countplot(x='age',hue='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.countplot(x='anaemia',hue='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.barplot(y='creatinine_phosphokinase',x='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.countplot(x='diabetes',hue='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.countplot(x='ejection_fraction',hue='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.countplot(x='high_blood_pressure',hue='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.barplot(y='platelets',x='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.countplot(x='serum_creatinine',hue='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.violinplot(y='serum_sodium',x='sex',hue='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.countplot(x='smoking',hue='DEATH_EVENT',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

sns.barplot(y='time',x='DEATH_EVENT',hue='sex',data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(15,6))

corr = df.corr()

sns.heatmap(corr,annot=True)
fig=plt.figure(figsize=(12,18))

for i in range (len(df.columns)):

    fig.add_subplot(9,4,i+1)

    sns.boxplot(y=df.iloc[:,i])

    

plt.tight_layout()

plt.show()

Q1_ejectionfraction=df['ejection_fraction'].quantile(.25)

Q3_ejectionfraction=df['ejection_fraction'].quantile(.75)

IQR_ejectionfraction=Q3_ejectionfraction-Q1_ejectionfraction

UpperLimit=Q3_ejectionfraction + 1.5*IQR_ejectionfraction

LowerLimit=Q1_ejectionfraction-1.5*IQR_ejectionfraction
print('lower',LowerLimit)

print('Upper',UpperLimit)

print('Q1',Q1_ejectionfraction)

print('Q3',Q3_ejectionfraction)
df['ejection_fraction']=np.where(df['ejection_fraction']>UpperLimit,UpperLimit,df['ejection_fraction'])
sns.boxplot(x='ejection_fraction',data=df)
df[df['serum_creatinine']>8]
df.drop(df[df['serum_creatinine']>8].index,inplace=True)
Q1=df['serum_creatinine'].quantile(.25)

Q3=df['serum_creatinine'].quantile(.75)

IQR=Q3-Q1

UpperLimit=Q3 + 1.5*IQR

LowerLimit=Q1-1.5*IQR
print('lower',LowerLimit)

print('Upper',UpperLimit)

print('Q1',Q1)

print('Q3',Q3)
df['serum_creatinine']=np.where(df['serum_creatinine']>UpperLimit,UpperLimit,df['serum_creatinine'])
sns.boxplot(x='serum_creatinine',data=df)
df.shape
df[df['serum_sodium']<120]
df.drop(df[df['serum_sodium']<120].index,inplace=True)
df.shape
Q1=df['serum_sodium'].quantile(.25)

Q3=df['serum_sodium'].quantile(.75)

IQR=Q3-Q1

UpperLimit=Q3 + 1.5*IQR

LowerLimit=Q1-1.5*IQR
print('lower',LowerLimit)

print('Upper',UpperLimit)

print('Q1',Q1)

print('Q3',Q3)
df['serum_sodium']=np.where(df['serum_sodium']<LowerLimit,LowerLimit,df['serum_sodium'])
sns.boxplot(x='serum_sodium',data=df)
df_modified = df[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]
df_modified
X = df_modified

Y=df['DEATH_EVENT']
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.3,random_state=0)
X_train.shape
X_test.shape
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()

model1.fit(X_train,Y_train)

Y_pred = model1.predict(X_test)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,Y_pred)

print('Accuracy is',"{:.2f}%".format(100*accuracy))
from sklearn.tree import DecisionTreeClassifier

model2 = DecisionTreeClassifier(criterion='entropy')

model2.fit(X_train,Y_train)

Y_pred = model2.predict(X_test)

accuracy = accuracy_score(Y_test,Y_pred)

print('Accuracy of Decision Tree model is',"{:.2f}%".format(100*accuracy))
from sklearn import *

model3=ensemble.RandomForestClassifier(n_estimators=150,criterion='entropy',

                                        random_state=0)

model3.fit(X_train,Y_train)



Y_pred = model3.predict(X_test)

accuracy = accuracy_score(Y_test,Y_pred)



print('Accuracy of Random Forest Model is:',"{:.2f}%".format(100*accuracy))
model_params = {'n_estimators':[140,145,150,155,160],

               'max_leaf_nodes':range(10,20),

               'criterion':['gini','entropy'],

                'max_depth':range(1,10),

               'min_impurity_decrease':[0.00005,0.0001,0.0002,0.0005,0.001,0.0015,0.002,0.005,0.01]}



rf_model_improved = ensemble.RandomForestClassifier(random_state=0)



random_search_object = model_selection.RandomizedSearchCV(rf_model_improved,model_params,

                                     n_iter=10,cv=10,random_state=0)



random_search_best_model = random_search_object.fit(X_train,Y_train)



Y_pred = random_search_best_model.predict(X_test)



accuracy= accuracy_score(Y_test,Y_pred)



print('Accuracy of Random Forest Model is:',"{:.2f}%".format(100*accuracy))

random_search_best_model.best_params_
from sklearn.neighbors import KNeighborsClassifier



model4 = KNeighborsClassifier(n_neighbors = 3, metric='euclidean')



model4.fit(X_train,Y_train)



Y_pred = model4.predict(X_test)



accuracy= accuracy_score(Y_test,Y_pred)



print('Accuracy of KNN Model is:',"{:.2f}%".format(100*accuracy))
model_params = {'leaf_size':range(1,50),

               'n_neighbors':range(1,30),

               'p':[1,2]}



knn_improved = KNeighborsClassifier()



grid_search_object = model_selection.GridSearchCV(knn_improved, model_params,cv=5)



grid_search_best_model = grid_search_object.fit(X_train,Y_train)



Y_pred = grid_search_best_model.predict(X_test)



accuracy= accuracy_score(Y_test,Y_pred)



print('Accuracy of KNN improved Model is:',"{:.2f}%".format(100*accuracy))

grid_search_best_model.best_params_
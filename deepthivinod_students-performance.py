

import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt#for data visualisation

import seaborn as sns#for data visualisation

%matplotlib inline

from sklearn.model_selection import train_test_split#for dividing the data set into test set and training set

from sklearn.ensemble import RandomForestClassifier#For building Random Forest model

from sklearn import metrics                     #for measuring performance of the model

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report#To see how well the predictions were classified correctly

from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
data.info()

data.gender=data.gender.astype('category')

data['race/ethnicity']=data['race/ethnicity'].astype('category')

data['parental level of education']=data['parental level of education'].astype('category')

data.lunch=data.lunch.astype('category')

data['test preparation course']=data['test preparation course'].astype('category')

#data.Result=data.Result.astype('category')

#data.Grade=data.Grade.astype('category')
data.isnull().sum()
print(data['parental level of education'].value_counts())

print(data.lunch.value_counts())

data['test preparation course'].value_counts()

data['race/ethnicity'].value_counts()

data.gender.value_counts()

print(data.lunch.unique())

print(data['parental level of education'].unique())

print(data['race/ethnicity'].unique())

print(data['test preparation course'].unique())



data.describe()


sns.countplot(data.gender) 
sns.countplot(x=data['race/ethnicity'])
sns.catplot(y='math score',x='gender',kind='bar',data=data)
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='parental level of education',y='math score',hue='gender',data=data)

ax.set_title('math score for different parental education and gender')

plt.show()  
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='parental level of education',y='reading score',hue='gender',data=data)

ax.set_title('math score for different parental education and gender ')

plt.show() 
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='parental level of education',y='writing score',hue='gender',data=data)

ax.set_title('math score for different parental education and gender')

plt.show() 


sns.catplot(y="parental level of education", hue="race/ethnicity", kind="count",

            palette="pastel", edgecolor=".6",

            data=data)
sns.catplot(y="race/ethnicity", hue="lunch", kind="count",

            palette="pastel", edgecolor=".6",

            data=data)
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='test preparation course',y='math score',hue='gender',data=data)

ax.set_title('math score based on test preparation')

plt.show() 
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='test preparation course',y='reading score',hue='gender',data=data)

ax.set_title('reading score based on test preparation')

plt.show() 
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='test preparation course',y='writing score',hue='gender',data=data)

ax.set_title('writing score based on test preparation')

plt.show() 
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='lunch',y='math score',hue='gender',data=data)

ax.set_title('math score based on lunch type')

plt.show() 
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='lunch',y='reading score',hue='gender',data=data)

ax.set_title('reading score based on lunch type')

plt.show() 
fig,ax=plt.subplots(figsize=(12,3))

sns.boxplot(x='lunch',y='writing score',hue='gender',data=data)

ax.set_title('writing score based on lunch type')

plt.show() 
data['Result'] = data.apply(lambda x : 'Fail' if x['math score'] <40 or 

                                    x['reading score'] < 40 or x['writing score'] <40 else 'Pass', axis =1)

data.head()
def  grade(s1,s2,s3,res):

  if res=='Fail':

    return 'F'

  elif sum([s1,s2,s3],0)/3 >=80  :

    return 'A'

  elif sum([s1,s2,s3],0)/3>=60 and sum([s1,s2,s3],0)/3<80:

    return 'B'

  elif sum([s1,s2,s3],0)/3 >=40 and sum([s1,s2,s3],0)/3<60:

    return 'C'

  else:

      return 'F'  
data['Grade'] = data.apply(lambda row: grade(row['math score'], row['reading score'],row['writing score'],row['Result']), axis=1)

                           

data.head()
sns.countplot(data['Grade'],order=['A','B','C','F'])
sns.catplot(y="parental level of education", hue="Result", kind="count",

            palette="pastel", edgecolor=".6",

            data=data)
data.columns

sns.catplot(y="Result", hue="test preparation course", kind="count",

            palette="pastel", edgecolor=".6",

            data=data)

data.Result=data.Result.astype('category')

data.Grade=data.Grade.astype('category')
X = data.drop('Grade',axis = 1)

scale=StandardScaler()

scale.fit(X.loc[:,'math score':'writing score'])

X.loc[:,'math score':'writing score']=scale.transform(X.loc[:,'math score':'writing score'])

X=pd.get_dummies(X,drop_first=True)

X.head()



y=data['Grade']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
rf_clf=RandomForestClassifier(n_estimators=100)

#y_train.shape

#X_train.shape

rf_clf.fit(X_train,y_train)

y_pred=rf_clf.predict(X_test)

metrics.accuracy_score(y_test,y_pred)

feature_imp=pd.Series(rf_clf.feature_importances_,index=X_train.columns).sort_values(ascending=False)

feature_imp
sns.barplot(x=feature_imp,y=feature_imp.index)

plt.title('Feature Importance')

plt.xlabel('score')

plt.ylabel('Features')



print(confusion_matrix(y_test,y_pred))



class_rep=classification_report(y_test,y_pred)

print(class_rep)
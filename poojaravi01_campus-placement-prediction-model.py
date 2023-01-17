import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn import metrics

import pandas as pd

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error
df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df.isna().sum()
df.drop('salary',axis=1,inplace=True)

df.drop('sl_no',axis=1,inplace=True)
df.head()
sns.pairplot(data=df,hue='status',palette='Set1')
sns.countplot(data=df,x='workex',hue='status',palette='Set2')
sns.countplot(data=df,x='gender',hue='status',palette='Set2')
sns.countplot(data=df,x='specialisation',hue='status',palette='Set2')
sns.countplot(data=df,x='degree_t',hue='status',palette='Set2')
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)

sns.distplot(df['ssc_p'],bins=50,)



plt.subplot(2,2,2)

sns.distplot(df['hsc_p'],bins=50,color='red')



plt.subplot(2,2,3)

sns.distplot(df['mba_p'],bins=50,color='green')



plt.subplot(2,2,4)

sns.distplot(df['etest_p'],bins=50,color='orange')
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),annot=True,linewidth=0.2)
df.drop('hsc_b',axis=1,inplace=True)

df.drop('ssc_b',axis=1,inplace=True)

df.drop('hsc_s',axis=1,inplace=True)
d1=pd.get_dummies(df['gender'],drop_first=True)

d2=pd.get_dummies(df['degree_t'],drop_first=True)

d3=pd.get_dummies(df['specialisation'],drop_first=True)

d4=pd.get_dummies(df['workex'],drop_first=True)

df=pd.concat([df,d1,d2,d3,d4],axis=1)

df.drop(['gender','workex','degree_t','specialisation'],axis=1,inplace=True)

labenc=LabelEncoder()

df['status']=labenc.fit_transform(df['status'])
df.head()
X=df.drop('status',axis=1)

y=df['status']
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=1)
sc=StandardScaler()

Xtrain=sc.fit_transform(Xtrain)

Xtest=sc.transform(Xtest)
logreg=LogisticRegression()

logreg.fit(Xtrain,ytrain)

ypred=logreg.predict(Xtest)

print('Accuracy is: {}%'.format(round(accuracy_score(ytest,ypred)*100,2)))
sns.distplot(ytest,bins=10,color='blue')

sns.distplot(ypred,bins=10,color='red')
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))
print(np.sqrt(mean_squared_error(ytest,ypred)))
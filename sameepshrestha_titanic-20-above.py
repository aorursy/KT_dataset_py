# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
%matplotlib inline


filepath1=('../input/titanic/train.csv')
filepath2=('../input/titanic/test.csv')
data_Train=pd.read_csv(filepath1)
data_Test=pd.read_csv(filepath2)
data_Train.head()
data_Train.info()
data_Train.isnull().sum()
data_Test.isnull().sum()
data_Train.describe()

plt.title('Train')
sns.countplot(data_Train['Survived'])
sns.countplot(data_Train['Pclass'],hue=data_Train['Survived'])
sns.distplot(data_Train[data_Train['Survived']==1]['Age'],bins=30,label='Survived')
sns.distplot(data_Train[data_Train['Survived']==0]['Age'],bins=30,color='red',label='Not survived')
plt.legend()

sns.distplot(data_Train[data_Train['Sex']=='male']['Age'],bins=30,label='male')
sns.distplot(data_Train[data_Train['Sex']=='female']['Age'],bins=30,color='red',label='female')
plt.legend()

sns.countplot(data_Train['Survived'],hue=data_Train['Sex'])
sns.countplot(data_Train['Embarked'],hue=data_Train['Survived'])

plt.subplot(121)
plt.title('female')
circle = plt.Circle((0, 0), 0.5, color = 'white')
data_Train[data_Train['Sex']=='female']['Survived'].value_counts().plot(kind='pie',figsize=(12, 12), rot=1, colors=[ '#1849CA','crimson'], autopct = '%.2f%%')
plots = plt.gcf()
plots.gca().add_artist(circle)
plt.legend()

plt.subplot(122)
plt.title('male')
circle = plt.Circle((0, 0), 0.5, color = 'white')
data_Train[data_Train['Sex']=='male']['Survived'].value_counts().plot(kind='pie',figsize=(12, 12), rot=1, colors=['crimson','#1849CA'], autopct = '%.2f%%')
plots = plt.gcf()
plots.gca().add_artist(circle)
plt.legend()
sns.pairplot(data_Train)
data_Train['Sex'].replace(['male','female'],[1,0],inplace=True)
data_Test['Sex'].replace(['male','female'],[1,0],inplace=True)
plt.subplots(figsize=(12, 10))
sns.heatmap(data_Train.corr().abs(),annot=True,cmap='Blues')
data_Train['Embarked'].fillna("S", inplace = True) 
data_Train['Embarked'].unique()
data_Train['Family']=data_Train['SibSp']+data_Train['Parch']+1
data_Test['Family']=data_Test['SibSp']+data_Test['Parch']+1

data_Train['Title']=data_Train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
data_Test['Title']=data_Test['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
print(data_Train['Title'].value_counts())
title=['Mr','Miss','Mrs','Master']
print(title)
data_Train['Title'] = np.where(data_Train['Title'].isin(title), data_Train['Title'], 'Other')
data_Test['Title'] = np.where(data_Test['Title'].isin(title), data_Test['Title'], 'Other')
data_Train['Title'].value_counts()
data_Train=pd.get_dummies(data_Train,columns=['Embarked'], prefix='embarked')
data_Test=pd.get_dummies(data_Test,columns=['Embarked'],prefix='embarked')
data_Train=pd.get_dummies(data_Train,columns=['Title'], prefix='Title')
data_Test=pd.get_dummies(data_Test,columns=['Title'],prefix='Title')
data_Train.head()
data_Train.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
data_Test.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

imp_mean=SimpleImputer(missing_values=np.nan,strategy='median')
imp_mean.fit(data_Train[['Age']])
data_Train['Age']=imp_mean.transform(data_Train[['Age']])
data_Test['Age']=imp_mean.transform(data_Test[['Age']])
imp_mean.fit(data_Test[['Fare']])
data_Test['Fare']=imp_mean.transform(data_Test[['Fare']])
Y_Train=data_Train['Survived']
X_Train=data_Train.drop(['Survived'],axis=1)
print(X_Train.head())
print(data_Test.head())

scaler=StandardScaler()
scaler.fit(X_Train)
X_Train=scaler.transform(X_Train)
X_Test=scaler.transform(data_Test)
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,make_scorer
X_train,X_test,Y_train,Y_test=train_test_split(X_Train,Y_Train,test_size=0.2, random_state=45)
print('Positive cases % in validation set: ', round(100 * len(Y_test[Y_test == 1]) / len(Y_test), 3), '%')
print('Positive cases % in train set: ', round(100 * len(Y_train[Y_train == 1]) / len(Y_train), 3), '%')
support=SVC()
support.fit(X_train,Y_train)
pred=support.predict(X_test)
accuracy = cross_val_score(support, X_train, Y_train, scoring='accuracy', cv = 5)
print(accuracy.mean())
sns.heatmap(confusion_matrix(pred,Y_test),annot=True,cbar=False)
f1_score(pred,Y_test)
support1=SVC(C=1,kernel='poly',degree=3,gamma=.05)
support1.fit(X_train,Y_train)
pred=support1.predict(X_test)
print(f1_score(pred,Y_test))
accuracy = cross_val_score(support1, X_train, Y_train, scoring='accuracy', cv = 5)
print(accuracy.mean())
sns.heatmap(confusion_matrix(pred,Y_test),annot=True,cbar=False)

from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold
str=StratifiedKFold(n_splits=4, shuffle = True, random_state = 999)
param_grid = {'C': [0.1, 1,5, 10, 100, ],  
              'gamma': [1, 0.1, 0.01,.05 ,0.001 ], 
              'kernel': ['rbf','linear','poly'],'degree':[2,3,4]}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,cv=str) 
grid.fit(X_train, Y_train) 
print(grid.best_params_) 
print(grid.best_estimator_) 
pred1=support1.predict(X_Test)
file='../input/titanic/gender_submission.csv'
submission=pd.read_csv(file)
submission['Survived']=pred1
submission[submission['Survived']==0].count()

#submission.to_csv('submission2.csv', index=False, encoding='utf-8')
from sklearn.model_selection import GridSearchCV  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

str=StratifiedKFold(n_splits=4, shuffle = True, random_state = 999)
rfc=RandomForestClassifier()
parameters={'n_estimators':[100,300,400,500,1000],'max_features':['auto','sqrt','log2'],
            'max_depth':[3,4,5,6,8],'criterion':['gini']}
CV_r = GridSearchCV(estimator=rfc, param_grid=parameters, cv= str,verbose=3)
CV_r.fit(X_Train, Y_Train)
print(CV_r.best_params_) 
print(CV_r.best_estimator_) 
rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 100, max_depth=4, criterion='gini')
rfc1.fit(X_train,Y_train)
pred=rfc1.predict(X_test)
print(f1_score(pred,Y_test))
accuracy = cross_val_score(rfc1, X_train, Y_train, scoring=make_scorer(f1_score), cv = 5)
print(accuracy.mean())
sns.heatmap(confusion_matrix(pred,Y_test),annot=True,cbar=False)
pred3=rfc1.predict(X_Test)
file='../input/titanic/gender_submission.csv'
submission3=pd.read_csv(file)
submission3['Survived']=pred3
#submission3.to_csv('submission3.csv', index=False, encoding='utf-8')
submission3[submission3['Survived']==0].count()

X_train.shape[1]
Train=X_train.copy()
Test=X_test.copy()
Train.shape
Y_train.head()
Y_t=Y_train.to_numpy()
from keras import Sequential 
from keras.layers import Dense
model = Sequential()
model.add(Dense(17, input_dim=15, activation='relu'))
model.add(Dense(17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Train, Y_t, validation_data=(Test,Y_test) ,epochs=250)
pred4=model.predict(Test)
pred4=(pred4>.5).astype(int)
pred4.reshape((1,179))

print(f1_score(pred4,Y_test))
sns.heatmap(confusion_matrix(pred4,Y_test),annot=True,cbar=False)
print(X_Test.shape)
pred5=model.predict(X_Test)
pred5=(pred5>.5).astype(int)
pred5=pred5.reshape((1,418))
pred5=np.squeeze(pred5)
pred5

file='../input/titanic/gender_submission.csv'
submission4=pd.read_csv(file)
submission4['Survived']=pred5
#submission4.to_csv('submission4.csv', index=False, encoding='utf-8')
submission4[submission4['Survived']==0].count()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

lm=LogisticRegression()
str=StratifiedKFold(n_splits=4, shuffle = True, random_state = 999)
param={'C':[.01,.1,.3,.5,1,2,10],'penalty':['l2']}
CV_r = GridSearchCV(estimator=lm, param_grid=param, cv= str,verbose=3)
CV_r.fit(X_Train, Y_Train)
print(CV_r.best_params_) 
print(CV_r.best_estimator_) 

lm=LogisticRegression(penalty='l2',solver='lbfgs',C=.1)
lm.fit(X_train,Y_train)
pred6=lm.predict(X_test)
print(f1_score(pred6,Y_test))
accuracy = cross_val_score(lm, X_train, Y_train, scoring=make_scorer(f1_score), cv = 5)
print(accuracy.mean())
sns.heatmap(confusion_matrix(pred6,Y_test),annot=True,cbar=False)
import xgboost
xgb=xgboost.XGBClassifier(colsample_bytree=0.3,eta=0.1,gamma=0.1)
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[('lr', lm), ('rf', rfc1), ('svm', support1),('xgb',xgb)], voting='hard')
eclf1.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
for clf in (lm,rfc1,support1,xgb,eclf1):
    clf.fit(X_train,Y_train)
    y_pred=clf.predict(X_test)
    print(clf.__class__.__name__,accuracy_score(Y_test,y_pred))

result=eclf1.predict(X_Test)
file='../input/titanic/gender_submission.csv'
submission5=pd.read_csv(file)
submission5['Survived']=result
submission5.to_csv('first.csv', index=False, encoding='utf-8')
submission5[submission5['Survived']==0].count()

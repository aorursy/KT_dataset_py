%%time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%%time
df_train=pd.read_csv('../input/train.csv')
%%time
df_test=pd.read_csv('../input/test.csv')
print(df_train.shape)
print(df_test.shape)
df_train.head(3)
df_train.dtypes
df_train['Survived'].value_counts()
%%time
df_train_test=[df_train, df_test]#combining train and test dataset
for data in df_train_test:
    data['Title'] = data['Name'].str.extract('([A-za-z]+)\.', expand=False)
df_train['Title'].value_counts()
df_test['Title'].value_counts()
title_mapping={"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Dr":3, "Rev":3, "Col":3, "Ms":3, "Dona":3, "Major":3, "Mme":3, "Don":3,
             "Sir":3, "Jonkheer":3, "Capt":3, "Lady":3, "Dona":3, "Mlle":3, "Countess":3 }
for data in df_train_test:
    data['Title']=data['Title'].map(title_mapping)
    
df_train.head()
df_test.head(4)
#delete unnecessary feature from dataset
df_train.drop('Name',axis=1, inplace=True)
df_test.drop('Name',axis=1, inplace=True)
gender_mapping={"male":0, "female":1}
for data in df_train_test:
    data['Sex']=data['Sex'].map(gender_mapping)
df_train.head(10)
df_train.tail(10)
df_train["Age"].fillna(df_train.groupby("Title")["Age"].transform("median"), inplace=True)
df_test["Age"].fillna(df_test.groupby("Title")["Age"].transform("median"), inplace=True)
for data in df_train_test:
    data.loc[data['Age'] <= 16, 'Age'] = 0,
    data.loc[(data['Age'] > 16) & (data['Age'] <= 26),'Age'] = 1,
    data.loc[(data['Age'] > 26) & (data['Age'] <= 36),'Age'] = 2,
    data.loc[(data['Age'] > 36) & (data['Age'] <= 62),'Age'] = 3,
    data.loc[data['Age'] > 62,'Age']= 4
df_train.head()
df_test.head()
for data in df_train_test:
    data['Embarked'] = data['Embarked'].fillna('S')
df_train.head()
embarked_mapping = {"S" : 0, "C" : 1, "Q" :2}
for data in df_train_test:
    data['Embarked'] = data['Embarked'].map(embarked_mapping)

#fill missing fare with median fare for each Pclass
df_train["Fare"].fillna(df_train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
df_test["Fare"].fillna(df_test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
for data in df_train_test:
    data.loc[data['Fare'] <= 17, 'Fare'] = 0,
    data.loc[(data['Fare'] > 17) & (data['Fare'] <= 30),'Fare'] = 1,
    data.loc[(data['Fare'] > 30) & (data['Fare'] <= 100),'Fare'] = 2,
    data.loc[data['Fare'] > 100,'Fare']= 3
df_train.head()
df_test.head()
for data in df_train_test:
    data['Cabin'] = data['Cabin'].str[:1]
cabin_mapping = { "A":0, "B": 0.4, "C":0.8, "D": 1.2, "E":1.6, "F":2, "G":2.4, "T":2.8 }
for data in df_train_test:
    data['Cabin'] = data['Cabin'].map(cabin_mapping)
#fill missing fare with median fare for each pclass
df_train['Cabin'].fillna(df_train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
df_test['Cabin'].fillna(df_test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
df_train['Family'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['Family'] = df_test['SibSp'] + df_test['Parch'] + 1
df_train['Family'].value_counts()
family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}
for data in df_train_test:
    data['Family'] = data['Family'].map(family_mapping)

df_train.head()
df_test.head()
features_remove = ['Ticket', 'SibSp','Parch']
df_train = df_train.drop(features_remove, axis=1)
df_test = df_test.drop(features_remove, axis=1)
df_train = df_train.drop(['PassengerId'], axis=1)
print("Train Set","\n",df_train.head(),"\n")
print("Test Set","\n",df_test.head())
df_train.isnull().sum()
df_target = df_train['Survived'] 
df_target.head()
df_train = df_train.drop(['Survived'], axis=1)
df_train.shape,df_target.shape
df_train.columns
df_train.head()
df_train.info()
%%time
# Importing Classifier Modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold= KFold(n_splits=10, shuffle=True, random_state=0)
clf = DecisionTreeClassifier()
clf
scoring = 'accuracy'
score = cross_val_score(clf, df_train, df_target, cv=k_fold, scoring = scoring)
print(score)
score_clf=round(np.mean(score)*100, 2)
score_clf
print(df_train.sample(3),"\n")
x_train=df_train
print(df_target.sample(3))
y_train=df_target

%%time
from datetime import datetime
start = datetime.now() 
clf.fit(x_train, y_train)
stop = datetime.now() 
print(clf)
execution_time_clf = stop-start 
print(execution_time_clf)
from sklearn.metrics import confusion_matrix
predicted_ytest=clf.predict(x_train)
conf_m=confusion_matrix(predicted_ytest,y_train)
conf_m
from sklearn.metrics import classification_report
report = classification_report(predicted_ytest,y_train)
print(report)    
rf=RandomForestClassifier(n_estimators=100)
rf
scoring = 'accuracy'
score = cross_val_score(rf, df_train, df_target, cv=k_fold, scoring = scoring)
print(score)
score_rf=round(np.mean(score)*100, 2)
score_rf
%%time
from datetime import datetime
start = datetime.now() 
rf.fit(x_train, y_train)
stop = datetime.now() 
execution_time_rf = stop-start 
print(execution_time_rf)
predicted_ytest=rf.predict(x_train)
conf_m=confusion_matrix(predicted_ytest,y_train)
conf_m
from sklearn.metrics import classification_report
report = classification_report(predicted_ytest,y_train)
print(report)  
svc=SVC(kernel='poly',random_state=10,decision_function_shape='ovo')
svc
scoring = 'accuracy'
score = cross_val_score(svc, df_train, df_target, cv=k_fold, scoring = scoring)
print(score)
score_svc=round(np.mean(score)*100, 2)
score_svc
%%time
start = datetime.now()
svc.fit(x_train, y_train)
stop = datetime.now()
print(svc)
execution_time_svc = stop-start 
print(execution_time_svc)
predicted_ytest=svc.predict(x_train)
conf_m=confusion_matrix(predicted_ytest,y_train)
conf_m
from sklearn.metrics import classification_report
report = classification_report(predicted_ytest,y_train)
print(report)  
from sklearn.svm import LinearSVC
lsvc = LinearSVC(random_state=10)
lsvc
scoring = 'accuracy'
score = cross_val_score(svc, df_train, df_target, cv=k_fold, scoring = scoring)
print(score)
score_lsvc=round(np.mean(score)*100, 2)
score_lsvc
start = datetime.now()
lsvc.fit(x_train,y_train)
stop = datetime.now()
execution_time_lsvc = stop-start 
print(execution_time_lsvc)
predicted_ytest=lsvc.predict(x_train)
conf_m=confusion_matrix(predicted_ytest,y_train)
conf_m
from sklearn.metrics import classification_report
report = classification_report(predicted_ytest,y_train)
print(report)  
comparison_dict = {'score':(score_clf,score_rf,score_svc,score_lsvc),'execution time':(execution_time_clf,execution_time_rf,execution_time_svc,execution_time_lsvc)}
comparison_dict
#Creating a dataframe ‘comparison_df’  
comparison_df =pd.DataFrame(comparison_dict) 
comparison_df.index= ['Decision Tree','Random Forest','SVC','LinearSVC'] 
comparison_df
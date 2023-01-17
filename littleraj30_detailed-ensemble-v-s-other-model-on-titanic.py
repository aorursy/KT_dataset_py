%%time

import numpy as np 

import pandas as pd

import warnings 

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns
train=pd.read_csv("../input/train.csv")
train.head()

train.describe()
train.describe(include=['O'])
train.hist(figsize=(10,8))
train.info()
train.Age.hist(bins=20)
sns.boxplot(x='Pclass',y='Age',data=train,hue='Survived')
print (train.groupby(['Pclass']).get_group(1).Age.mean())

print (train.groupby(['Pclass']).get_group(2).Age.mean())

print (train.groupby(['Pclass']).get_group(3).Age.mean())
train['Age']=train.groupby(['Pclass','Survived'])['Age'].transform(lambda x:x.fillna(x.mean()))
train.info()
sns.stripplot(y='Fare',x='Pclass',hue='Survived',data=train)
train.groupby(['Pclass','Survived'])['Fare'].mean()
plt.hist(train.Fare,bins=30)

plt.xlabel('Fare')

plt.ylabel('count')
train.Fare=np.ceil(train.Fare)

train['fare']=pd.cut(train.Fare,bins=[0,8,13,20,30,50,80,600],labels=['a','b','c','d','e','f','g'],right=False)
sns.countplot(x='fare',hue='Survived',data=train)
sns.countplot(x='SibSp',hue='Survived',data=train)
sns.countplot(x='Parch',hue='Survived',data=train)
train['members']=train['SibSp']+train['Parch']
sns.countplot(x='members',hue='Survived',data=train)
train.members.value_counts()
train[train.members>6].Survived.value_counts()

train.members.replace({10:7},inplace=True)
train.head()
attributes=['Survived','Pclass','Sex','Age','Embarked','fare','members']
train=train[attributes]
train.head()
sns.countplot(x='Embarked',hue='Survived',data=train)
train[train.Embarked.isnull()]
sns.catplot(kind='point',x='Embarked',y='Pclass',hue='Sex',data=train)
train.groupby(['Pclass','Sex']).get_group((1,'female')).Embarked.value_counts()
train.Embarked.fillna('C',inplace=True)
train.info()
def func(x):

    if(x.dtype=='O'):

        x=x.astype('category')

    return(x)
train=train.apply(func,axis=0)
train.info()
train.members=train.members.astype('category')

train.Survived=train.Survived.astype('category')

train.Pclass=train.Pclass.astype('category')

train.Age=train.Age.astype('int64')
train.info()
df_label=train.Survived

del train['Survived']

df=pd.get_dummies(train)
from sklearn.preprocessing import StandardScaler
scaled=StandardScaler().fit_transform(df)

df=pd.DataFrame(scaled,index=df.index,columns=df.columns)

df=pd.concat([df,df_label],axis=1)
df.head()
train=df

train.shape
index=np.random.permutation(891)

train=train.loc[index,:]

train.shape
train_label=train.Survived

del train['Survived']
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier

sgd=SGDClassifier(n_iter=200,penalty='l1',epsilon=1e-20,random_state=8349)

score=cross_val_predict(sgd,train,train_label,cv=10)

confusion_matrix(train_label,score)
from sklearn.metrics import accuracy_score

acc_lc=accuracy_score(train_label,score)

acc_lc
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state=73289471,class_weight='balanced')

score=cross_val_predict(lr,train,train_label,cv=10)

confusion_matrix(train_label,score)
from sklearn.metrics import accuracy_score

acc_lr=accuracy_score(train_label,score)

acc_lr
from sklearn.neighbors import KNeighborsClassifier

value=[]

for k in range(1,20):

    knn=KNeighborsClassifier(k,algorithm='brute')

    score=cross_val_predict(knn,train,train_label,cv=10)

    value.append(accuracy_score(train_label,score))
df=pd.DataFrame(value,index=range(1,20),columns=['accuracy'])
df.set_index='K value'

df.sort_values(ascending=False,by='accuracy')
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(7,algorithm='brute')

score=cross_val_predict(knn,train,train_label,cv=10)

acc_knn=accuracy_score(train_label,score)

acc_knn
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier(random_state=1341)

score=cross_val_predict(dtc,train,train_label,cv=10)

confusion_matrix(train_label,score)
acc_dtc=accuracy_score(train_label,score)

acc_dtc
from sklearn.svm import SVC

svm=SVC(kernel='rbf',C=20,gamma=0.05,random_state=2317)

score=cross_val_predict(svm,train,train_label,cv=10)

confusion_matrix(train_label,score)
acc_svm=accuracy_score(train_label,score)

acc_svm
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=200,random_state=167123)

score=cross_val_predict(rf,train,train_label,cv=10)

confusion_matrix(train_label,score)
acc_rf=accuracy_score(train_label,score)

acc_rf
from sklearn.ensemble import ExtraTreesClassifier

etc=ExtraTreesClassifier(n_estimators=200,random_state=67)

score=cross_val_predict(etc,train,train_label,cv=10)

confusion_matrix(train_label,score)
acc_etc=accuracy_score(train_label,score)

acc_etc
from sklearn.ensemble import AdaBoostClassifier

ada=AdaBoostClassifier(dtc,n_estimators=200,

algorithm='SAMME.R',learning_rate=0.01,random_state=13247)

score=cross_val_predict(ada,train,train_label,cv=10)

confusion_matrix(train_label,score)
acc_ada=accuracy_score(train_label,score)

acc_ada
from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier(n_estimators=200,learning_rate=0.01,random_state=11233)

score=cross_val_predict(gb,train,train_label,cv=10)

confusion_matrix(train_label,score)
acc_gb=accuracy_score(train_label,score)

acc_gb
from sklearn.ensemble import BaggingClassifier

bp=BaggingClassifier(SVC(kernel='rbf',C=20,gamma=0.05,random_state=87),n_estimators=200, bootstrap=False ,

                     n_jobs=-1,random_state=82139 )

score=cross_val_predict(bp,train,train_label,cv=10)

confusion_matrix(train_label,score)
acc_bp=accuracy_score(train_label,score)

acc_bp
df=pd.DataFrame([acc_lc*100,acc_lr*100,acc_knn*100,acc_dtc*100,acc_svm*100,acc_rf*100,

             acc_etc,acc_ada*100,acc_gb*100,acc_bp*100],

            index=['Linear Classifier','Logistic','KNN','Decision Tree','SVM','Random Forest',

                  'Extra Trees','ADA boost','Gradient Boost','Bagging and pasting'],columns=['Accuracy'])
df=df.sort_values(ascending=False,by='Accuracy')
color=sns.color_palette

sns.barplot(data=df, y=df.index,x='Accuracy')

#plt.xticks(rotation=90)
from sklearn.model_selection import RandomizedSearchCV

def r_search(classifier,param,data,data_label,fold):

    rs=RandomizedSearchCV(classifier,param_distributions=param,cv=fold,n_jobs=-1)

    rs.fit(data,data_label)

    return(rs.best_params_ , rs.best_score_, rs.best_estimator_)
param={'max_features':[7,9,13,],'max_depth':[5,7,9,12],'min_samples_split':[25,40,55],

       'min_samples_leaf':[3,5,13,23],'max_leaf_nodes':[3,7,13,19],

      'n_estimators':[100,200,500,1000],'learning_rate':[1,0.1,0.01,0.001]}

best_param , best_score , best_estimator= r_search(GradientBoostingClassifier(random_state=9248309),

                                 param,train,train_label,10)
print(best_param,'\n' ,best_score)
gb=best_estimator
from sklearn.ensemble import VotingClassifier

vc=VotingClassifier(estimators=[('rf',svm),('gb',gb),

                                ('svm',lr)],voting='hard')

score=cross_val_predict(vc,train,train_label,cv=10)

confusion_matrix(train_label,score)
acc_vc=accuracy_score(train_label,score)

acc_vc
gb.fit(train,train_label)
test=pd.read_csv('../input/test.csv')
test.head()
attributes=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

test=test[attributes]
test.info()
test['Age']=test.groupby('Pclass')['Age'].transform(lambda x:x.fillna(x.mean()))
test.Embarked.fillna('C',inplace=True)
test.info()
test['members']=test['SibSp']+test['Parch']

del test['SibSp']

del test['Parch']
test.members.replace({10:7},inplace=True)
test.Fare=np.ceil(test.Fare)

test['fare']=pd.cut(test.Fare,bins=[0,8,13,20,30,50,80,600],labels=['a','b','c','d','e','f','g'],right=False)
test.members=test.members.astype('category')

test.Pclass=test.Pclass.astype('category')

test.Age=test.Age.astype('int64')

test.fare=test.fare.astype('category')

test.Embarked=test.Embarked.astype('category')
test.info()
test.fare.value_counts()
test.fare.fillna('b',inplace=True)

test.info()
del test['Fare']
test.head()
test=pd.get_dummies(test)
scaled=StandardScaler().fit_transform(test)

test=pd.DataFrame(scaled,index=test.index,columns=test.columns)

test.head()
test.shape
prediction=gb.predict(test)
sample=pd.read_csv('../input/gender_submission.csv')
sample.head()
s=pd.DataFrame({'PassengerId':sample.PassengerId,'Survived':prediction})

s.head()
s.to_csv('submission.csv',index=False)
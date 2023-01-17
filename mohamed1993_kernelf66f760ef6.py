# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

passenger_id_test=test_data['PassengerId']

passenger_id_train=train_data['PassengerId']
train_data.isna().sum()

test_data.isna().sum()
######drop unusefull column

train_data=train_data.drop(columns=['Ticket','Cabin','Name'])

test_data=test_data.drop(columns=['Ticket','Cabin','Name'])
#######convert 

test_data=pd.get_dummies(data=test_data)

train_data=pd.get_dummies(data=train_data)
test_data=pd.get_dummies(data=test_data,columns=['Pclass'])

train_data=pd.get_dummies(data=train_data,columns=['Pclass'])
train_data_age_train=train_data[train_data['Age'].notnull()]

y_train_train=train_data[train_data['Age'].notnull()]['Survived']

y_train_train2=train_data[train_data['Age'].isna()]['Survived']

train_data_age_fit=train_data[train_data['Age'].isna()]

y_train_data_age_train=train_data_age_train['Age']

y_train=train_data['Survived']

train_data_age_train=train_data_age_train.drop(columns=['Age','Survived'])

train_data_age_fit=train_data_age_fit.drop(columns=['Age','Survived'])

train_data_age_train.keys()
from sklearn.ensemble import GradientBoostingRegressor

gradient=GradientBoostingRegressor()

gradient.fit(train_data_age_train,y_train_data_age_train)

y_age=gradient.predict(train_data_age_train)

filled_age=gradient.predict(train_data_age_fit)

filled_age=pd.DataFrame(filled_age,columns=['Age'])

filled_age=filled_age.reset_index().drop(columns=['index'])

train_data_age_fit=train_data_age_fit.reset_index().drop(columns=['index'])

train_data_age_train['Age']=y_train_data_age_train

train_data_age_fit['Age']=filled_age['Age']



train_data=pd.concat([train_data_age_train,train_data_age_fit],axis=0)

y_train=pd.concat([y_train_train,y_train_train2],axis=0)

test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].median())
test_is=test_data[test_data['Age'].isna()]

test_not=test_data[test_data['Age'].notnull()]
test_is=test_is.drop(columns=['Age'])

age_test=gradient.predict(test_is)

age_test=pd.DataFrame(age_test,columns=['Age']).reset_index().drop(columns='index')

test_is=test_is.reset_index().drop(columns=['index'])

test_is['Age']=age_test

test_data=pd.concat([test_is,test_not],axis=0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

rf=RandomForestClassifier()
####################################

logi_clf = LogisticRegression(C=5,penalty='l1',random_state=0)

logi_clf.fit(train_data,y_train)

p1=logi_clf.predict_proba(train_data)[:,1]

###########################

svm_clf = SVC(kernel='poly',C=100,random_state=0,degree=3,gamma='scale',probability=True)

svm_clf.fit(train_data,y_train)

p2=svm_clf.predict_proba(train_data)[:,1]

###############################

dt_clf = DecisionTreeClassifier(criterion='gini',random_state=0)

dt_clf.fit(train_data,y_train)

p3=dt_clf.predict_proba(train_data)[:,1]

######################################

knn_clf = KNeighborsClassifier(n_neighbors=10,weights='uniform',p=1)

knn_clf.fit(train_data,y_train)

p4=knn_clf.predict_proba(train_data)[:,1]

#################################################

gnb_clf = GaussianNB()

gnb_clf.fit(train_data,y_train)

p5=gnb_clf.predict_proba(train_data)[:,1]

ada=AdaBoostClassifier()

ada.fit(train_data,y_train)

p6=ada.predict_proba(train_data)[:,1]

gradient=GradientBoostingClassifier()

gradient.fit(train_data,y_train)

p7=gradient.predict_proba(train_data)[:,1]

rf=RandomForestClassifier()

rf.fit(train_data,y_train)

p8=rf.predict_proba(train_data)[:,1]
p1=p1.reshape(-1,1)

p2=p2.reshape(-1,1)

p3=p3.reshape(-1,1)

p4=p4.reshape(-1,1)

p5=p5.reshape(-1,1)

p6=p6.reshape(-1,1)

p7=p7.reshape(-1,1)

p8=p8.reshape(-1,1)

last=np.hstack((p1,p2,p3,p4,p5,p6,p7,p8))

last=pd.DataFrame(data=last)

last
ada_1=AdaBoostClassifier()

ada_1.fit(last,y_train)

y_last=ada_1.predict(last)

from sklearn.metrics import accuracy_score

accuracy_score(y_train,y_last)
test_data = test_data[['PassengerId', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male','Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1', 'Pclass_2','Pclass_3','Age']]

##################################

p1=logi_clf.predict_proba(test_data)[:,1]

###########################

p2=svm_clf.predict_proba(test_data)[:,1]

###############################



p3=dt_clf.predict_proba(test_data)[:,1]

######################################

p4=knn_clf.predict_proba(test_data)[:,1]

#################################################



p5=gnb_clf.predict_proba(test_data)[:,1]

p6=ada.predict_proba(test_data)[:,1]



p7=gradient.predict_proba(test_data)[:,1]



p8=rf.predict_proba(test_data)[:,1]
p1=p1.reshape(-1,1)

p2=p2.reshape(-1,1)

p3=p3.reshape(-1,1)

p4=p4.reshape(-1,1)

p5=p5.reshape(-1,1)

p6=p6.reshape(-1,1)

p7=p7.reshape(-1,1)

p8=p8.reshape(-1,1)



last=np.hstack((p1,p2,p3,p4,p5,p6,p7,p8))

last=pd.DataFrame(data=last)

last
#from sklearn.ensemble import VotingClassifier



#eclf2 = VotingClassifier(estimators=[('LogisticRegression', logi_clf), ('svc', svm_clf), ('DecisionTreeClassifier', dt_clf),('knn_clf',knn_clf),('GaussianNB',gnb_clf),('ada',ada),('gradient',gradient),('rf',rf)],voting='soft')

#eclf2.fit(train_data, y_train)

#we=eclf2.predict(train_data)

#from sklearn.metrics import accuracy_score

#we=eclf2.predict(test_data)

#accuracy_score(y_train,we)



y_last_test=ada_1.predict(last)

my_submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_last_test})
my_submission.to_csv('submission.csv', index=False)
my_submission
train_data.head()
test_data.head()
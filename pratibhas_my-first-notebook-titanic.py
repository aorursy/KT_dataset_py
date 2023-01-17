# import libraries which are needed for data analysis

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

#laod data and combine

train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')
train.shape, test.shape
#train data has 891 passenger information whereas test data has 418. Let's combine these two into dataset.

dataset=pd.concat(objs=[train,test],axis=0).reset_index(drop=True)

dataset.head(3)
testid=test['PassengerId']
#check out the missing values in feature

dataset.isnull().sum()
#correlation in dataset

sns.heatmap(train.corr(),annot=True,cmap="YlGnBu")
len(train['PassengerId'].unique())
dataset=dataset.drop('PassengerId',axis=1)
dataset.loc[dataset['Embarked'].isnull()]
dataset.groupby(['Embarked','Pclass'])['Fare'].median()
dataset['Embarked']=dataset['Embarked'].fillna('C')
sns.countplot(x='Embarked',hue='Survived',data=dataset)
sns.countplot(x='Embarked',hue='Pclass',data=dataset)
dataset.loc[dataset['Fare'].isnull()]
dataset['Fare']=dataset['Fare'].fillna(8.05)
dataset.loc[dataset['Fare']==0.0]
len(dataset.loc[dataset['Fare']==0.0])
#list of indices where fare=0.

inde=[dataset.loc[dataset['Fare']==0.0].index] 

for i in inde:

    dataset.loc[i,'Fare']=dataset.groupby(['Embarked','Pclass'])['Fare'].transform(lambda x : x.median())

sns.kdeplot(dataset['Fare'])
dataset['Fare']=np.log(dataset['Fare'])
sns.kdeplot(dataset['Fare'])
f,axes=plt.subplots(2,1,figsize=(7,8))

a=sns.kdeplot(train['Age'][(train['Survived']==0) & train['Age'].notnull()],ax=axes[0])

a=sns.kdeplot(train['Age'][(train['Survived']==1) & train['Age'].notnull()],ax=axes[0])

a.legend(['Dead','Survived'])

a=sns.kdeplot(train['Age'][(train['Sex']=='male') & train['Age'].notnull()],ax=axes[1])

a=sns.kdeplot(train['Age'][(train['Sex']=='female') & train['Age'].notnull()],ax=axes[1])

a.legend(['male','female'])
dataset['title']=dataset['Name'].str.extract('([A-Za-z]+)\.')
dataset['title'].value_counts()
dataset['title']=dataset['title'].replace('Mlle','Miss')

dataset['title']=dataset['title'].replace('Mme','Mrs')

dataset['title']=dataset['title'].replace(['Rev','Dr','Col',

                                            'Major','Ms','Dona','Lady','Sir',

                                            'Jonkheer','Don','Capt','Countess'],

                                           'Rare')

dataset['title'].unique()
sns.boxplot(x='title',y='Age',hue='Pclass',data=dataset)
dataset.groupby(['Pclass','title'])['Age'].median()
dataset.loc[(dataset['Pclass']==3) & (dataset['title']=='Rare')]
dataset['Age']=np.where((dataset['Pclass']==3) & (dataset['title']=='Rare'),0,dataset['Age'])
dataset['Age']=dataset.groupby(['Pclass','title'])['Age'].transform(lambda x:x.fillna(x.median()))
dataset['Age']=np.where((dataset['Pclass']==3) & (dataset['title']=='Rare'),44.75,dataset['Age'])
dataset['Age'].isnull().sum()
sns.kdeplot(dataset['Age'])
dataset['Cabin']=dataset['Cabin'].fillna('X')

dataset['Cabin']=dataset['Cabin'].str.get(0)
dataset['Cabin'].unique()
sns.countplot(x='Cabin',hue='Pclass',data=dataset)
sns.countplot(x='Cabin',hue='Embarked',data=dataset)
sns.catplot(x='Cabin',y='Survived',data=dataset,kind='bar')
dataset.loc[dataset['Ticket']=='LINE']
ticket=[]

for i in dataset['Ticket']:

    if not i.isdigit():

        ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])

    elif i=='LINE':

        ticket.append('LINE')

    else:

        ticket.append('X')

dataset['Ticket']=ticket
sns.barplot(x='Sex',y='Survived',data=dataset)
dataset['Sex']=dataset['Sex'].map({'male':0,'female':1})
sns.barplot(x='Pclass',y='Survived',data=dataset)
sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=dataset,kind='bar')
dataset['family']=dataset['Parch']+dataset['SibSp']+1
sns.barplot(x='family',y='Survived',data=dataset)
dataset['alone']=np.where(dataset['family']==1,1,0)

dataset['small-family']=np.where(dataset['family']==2,1,0)

dataset['med-family']=np.where((dataset['family']>=3) & (dataset['family']<=4),1,0)

dataset['big-family']=np.where(dataset['family']>=5,1,0)
#Embarked, title, cabin ticket

dataset=pd.get_dummies(dataset,columns=['Embarked'],prefix='em',drop_first=True)

dataset=pd.get_dummies(dataset,columns=['title'],prefix='tit',drop_first=True)

dataset=pd.get_dummies(dataset,columns=['Cabin'],prefix='cab',drop_first=True)

dataset=pd.get_dummies(dataset,columns=['Ticket'],prefix='tick',drop_first=True)

dataset=pd.get_dummies(dataset,columns=['Pclass'],prefix='clas',drop_first=True)
dataset=dataset.drop(['Name'],axis=1)
dataset.head(3)
#separate data into train, test

train=dataset[:len(train)]

test=dataset[len(train):]
x_train=train.drop('Survived',axis=1)

y_train=train['Survived'].astype(int)

x_test=test.drop('Survived',axis=1)
# import libraries which are needed for modeling

from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier,VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold,GridSearchCV
fold=StratifiedKFold(10)
adaboost_param= {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :list(range(1,5,1)),

              "learning_rate":[0.0002,0.04,0.0007,0.1]

              }

dt=DecisionTreeClassifier()

ada=AdaBoostClassifier(dt,random_state=0)

grid_ada=GridSearchCV(ada,param_grid=adaboost_param,cv=fold,scoring='accuracy',verbose=1)

grid_ada.fit(x_train,y_train)
grid_ada.best_score_
ada_best=grid_ada.best_estimator_
rf_param = {"max_depth": [None],

              "max_features":[2,4,6],

              "min_samples_split":[4,6],

              "min_samples_leaf":[2,3,6],

              "bootstrap": [False],

              "n_estimators" :[100,150],

              "criterion": ["gini"]}

rf=RandomForestClassifier()

grid_rf=GridSearchCV(rf,param_grid = rf_param, cv=fold, scoring="accuracy",verbose = 1)

grid_rf.fit(x_train,y_train)
grid_rf.best_score_
rf_best=grid_rf.best_estimator_
#gradient boosting

gb_param= {'loss' : ["deviance"],

              'n_estimators' : [250,300],

              'learning_rate': [0.0001,0.001,0.05],

              'max_depth':[3,5],

              'min_samples_leaf':[100,150],

              'max_features': [0.1,0.5] 

              }

gradb=GradientBoostingClassifier()

grid_gradb= GridSearchCV(gradb,param_grid = gb_param, cv=fold, scoring="accuracy",verbose = 1)

grid_gradb.fit(x_train,y_train)
gradb_best=grid_gradb.best_estimator_
grid_gradb.best_score_
#correlation between predicted values for chosen classifiers

pred_ada=pd.Series(ada_best.predict(x_test),name='adaboost')

pred_rf=pd.Series(rf_best.predict(x_test),name='random-forest')

pred_gradb=pd.Series(gradb_best.predict(x_test),name='gradient-boost')

result=pd.concat([pred_ada,pred_rf,pred_gradb],axis=1)
sns.heatmap(result.corr(),annot=True)
vote=VotingClassifier(estimators=[('rf',rf_best),('grad',gradb_best)],voting='soft')

vote_result=vote.fit(x_train,y_train)

y_pred=vote_result.predict(x_test).astype(int)
sub=pd.DataFrame({'PassengerId':testid,'Survived':y_pred})

sub.to_csv('submission.csv',index=False)
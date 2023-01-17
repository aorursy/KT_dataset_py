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

import random as rnd 

import matplotlib.pyplot as plt
%matplotlib inline 
import seaborn as sns 

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
all_data = [train,test]
train.tail(10)
train.describe()
train.describe(include='O')
train.info()
test.head()
test.info()
#### ANALYSIS
def cor_map(df):
    cor=df.corr()
    _,ax=plt.subplots(figsize=(12,10))
    cmap = sns.diverging_palette(219,6,as_cmap=True)
    _=sns.heatmap(cor,cmap=cmap,annot=True,
                 cbar=True,ax=ax,square=True)

def dist(df,var,target,**kwargs):
    row = kwargs.get('row',None)
    col = kwargs.get('col',None)
    facet = sns.FacetGrid(df, hue=target,aspect=4,row=row,col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()

def category(df,cat,target,**kwargs):
    row=kwargs.get('row',None)
    col=kwargs.get('col',None)
    facet=sns.FacetGrid(df,row=row,col=col)
    facet.map(sns.barplot,cat,target)
    facet.add_legend()
cor_map(train.drop(['PassengerId'],axis=1))
train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',ascending=False)
dist(train,target='Survived',var='Fare')
dist(train,target='Survived',var='Fare',row='Pclass')
dist(train,target='Pclass',var='Fare')
dist(train,target='Survived',var='Age')
category(train,'Sex','Fare',row='Embarked',col='Survived')
category(train,'Sex','Survived')
train[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
category(train,'Embarked','Survived')
train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)
pd.crosstab(train.Survived,train.Cabin.isnull())
guess_ages = np.zeros((3,9))
for dataset in all_data:
    dataset['ageFill']=dataset.Age.isnull().map({False:0,True:1})
    med_all = dataset['Age'].median()
    for i in range(0,3):
        for j in range(0,9):
            guess_df=dataset[(dataset['Pclass']==i+1)&\
                            (dataset['SibSp']==j)]['Age'].dropna()
            age_guess=guess_df.median()
            try: 
                guess_ages[i,j] = int(age_guess/0.5+0.5)*0.5
            except:
                guess_ages[i,j]=med_all
    for i in range(0,3):
        for j in range(0,9):
            dataset.loc[(dataset.Age.isnull())&(dataset.Pclass==i+1)&(dataset.SibSp==j),'Age']=guess_ages[i,j]
    
    dataset['Age']=dataset['Age'].astype(int)
    print(guess_ages)
    
train.head(10)
freq_port = train.Embarked.dropna().mode()[0]
train.Embarked = train.Embarked.fillna(freq_port)
print(freq_port)
    
test['Fare']=test.Fare.fillna(test.Fare.mean())
test.info()
train.info()
for dataset in all_data:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'],train['Sex'])
for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don',\
                                                'Dr','Jonkheer','Lady','Major',\
                                                'Rev','Sir'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    
train[['Title','Survived']].groupby(['Title'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train.info()
dataset.Ticket.unique()
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'
for dataset in all_data:
    dataset[ 'ticketPos' ] = dataset[ 'Ticket' ].map( cleanTicket )
train.head()
train.ticketPos.unique()
for dataset in all_data:
    title_mapping = {'Mr':1,'Rare':2,'Master':3,'Miss':4,'Mrs':5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train.head()
for dataset in all_data:
    dataset['Sex']=dataset.Sex.map({'male':0,'female':1})
train.head()
for dataset in all_data:
    dataset['Embarked']=dataset.Embarked.map({'S':0,'Q':1,"C":2})
train.head()
for dataset in all_data:
    dataset['AgeBand']=pd.cut(dataset['Age'],5,labels=[0,1,2,3,4])
    dataset['AgeBand']=dataset.AgeBand.astype(int)
train[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='Survived',ascending=False)
for dataset in all_data:
    dataset['cabinRec']=dataset.Cabin.isnull().map({False:0,True:1})
train.head()
for dataset in all_data:
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 5,labels=[1,2,3,4,5])
    dataset['FareBin'] = dataset.FareBin.astype(int)

#     label = LabelEncoder()
#     dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin']).astype(int)
train.head()
# train=pd.concat([train,pd.get_dummies(train['ticketPos'],prefix='ticket')],axis=1)
# test=pd.concat([test,pd.get_dummies(test['ticketPos'],prefix='ticket')],axis=1)
train_df=train.drop(['AgeBand','FareBin','Ticket','Cabin','Name','PassengerId','ticketPos'],axis=1)
test_df=test.drop(['AgeBand','FareBin','Ticket','Cabin','Name','PassengerId','ticketPos'],axis=1)
# train_df=train.drop(['Ticket','Cabin','Ticket','Name','PassengerId','ticketPos'],axis=1)
# test_df=test.drop(['Ticket','Cabin','Ticket','Name','PassengerId','ticketPos'],axis=1)
# train_df=train_df.drop('FareBin',axis=1)
# test_df=test_df.drop('FareBin',axis=1)
combine_df = [train_df,test_df]
train_df.head()
for dataset in combine_df:
    dataset['Cabin_Class'] = (dataset.cabinRec+1)*dataset.Pclass
train_df=train_df.drop('Sex',axis=1)
test_df=test_df.drop('Sex',axis=1)
X_train_valid = train_df.drop('Survived',axis=1)
y_train_valid = train_df['Survived']
X_test = test_df
X_train,X_valid,y_train,y_valid=train_test_split(X_train_valid,y_train_valid,test_size=0.25,random_state=0)
print(X_train.shape,X_test.shape,X_valid.shape)
# DecisionTree classifier 
def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))
tree=DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
pred_tree = tree.predict(X_valid)
print(tree.score(X_train,y_train),tree.score(X_valid,y_valid))
plot_model_var_imp(tree,X_train,y_train)
rf_model = RandomForestClassifier(random_state=0,n_estimators=10,max_features=6,)
rf_model.fit(X_train,y_train)
pred_rf = rf_model.predict(X_valid)
print(rf_model.score(X_train,y_train),rf_model.score(X_valid,y_valid))
plot_model_var_imp(rf_model,X_train,y_train)
xgb = XGBClassifier(random_state=0,n_estimators=500,learning_rate=0.05,reg_lambda=20)
xgb.fit(X_train,y_train)
pred_xgb = xgb.predict(X_valid)
print(xgb.score(X_valid,y_valid))
plot_model_var_imp(xgb,X_train,y_train)
from sklearn.ensemble import GradientBoostingClassifier
gradb= GradientBoostingClassifier(learning_rate=0.01,random_state=0,n_estimators=2000,max_features=4)
gradb.fit(X_train,y_train)
print(gradb.score(X_valid,y_valid))
plot_model_var_imp(gradb,X_train,y_train)
from sklearn.metrics import make_scorer,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
clf = XGBClassifier(random_state=0,n_jobs=-1)
cv_sets = ShuffleSplit(X_train.shape[0], n_iter =5, test_size = 0.20, random_state = 7)
parameters = {'n_estimators':list(range(100,1000,100)),
#              'max_depth':np.linspace(1,32,32,endpoint=True,dtype=np.int),
             'learning_rate':[0.05,0.1,0.25,0.5,0.75],
             'reg_lambda':[1,10,15,20,25]}
acc_scorer=make_scorer(accuracy_score)
grid_obj=GridSearchCV(clf, parameters, scoring=acc_scorer,verbose=1,cv=cv_sets)
grid_obj= grid_obj.fit(X_train,y_train)
clf_best = grid_obj.best_estimator_
clf_best.fit(X_train,y_train)
print(clf_best.score(X_valid,y_valid))
plot_model_var_imp(clf_best,X_train,y_train)
ids=test['PassengerId']
predictions = clf_best.predict(X_test)


my_submission  = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
my_submission.to_csv('submission.csv', index=False)


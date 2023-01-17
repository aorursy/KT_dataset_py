import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection

train=pd.read_csv('../input/titanic/train (1).csv')
test=pd.read_csv('../input/titanic/test (1).csv')
train['Ticket_type']=train['Ticket'].apply(lambda x: x[0:3])
train['Ticket_type']=train['Ticket_type'].astype('category')
train['Ticket_type']=train['Ticket_type'].cat.codes

test['Ticket_type']=test['Ticket'].apply(lambda x: x[0:3])
test['Ticket_type']=test['Ticket_type'].astype('category')
test['Ticket_type']=test['Ticket_type'].cat.codes

raw_data=[train,test]
train.describe()

for d in raw_data:
    d['Family']=d['SibSp']+d['Parch']
    d['IsAlone']=0
    d.loc[d['Family']==0 , 'IsAlone']=1
    d['Sex']=d['Sex'].map({'female':0,'male':1})
    d['Sex']=d['Sex'].astype(int)
    d['Fare']=d['Fare'].fillna(train['Fare'].median())
    d['Embarked']=d['Embarked'].fillna('S')
    d['Embarked']=d['Embarked'].map({'S':0,'C':1,'Q':2})
    d['Embarked']=d['Embarked'].astype(int)
    age_mean=d['Age'].mean()
    age_std=d['Age'].std()
    null_age=d['Age'].isnull().sum()
    age_list=np.random.randint(age_mean-age_std,age_mean+age_std,size=null_age)
    d['Age'][np.isnan(d['Age'])]=age_list
    d['Age']=d['Age'].astype(int)
    d.loc[d['Age']<=16, 'Age']=0
    d.loc[d['Age']>16 & (d['Age']<=32), 'Age']=1
    d.loc[d['Age']>32 & (d['Age']<=48), 'Age']=2
    d.loc[d['Age']>48 & (d['Age']<=64), 'Age']=3
    d.loc[(d['Fare']<=7.91), 'Fare']=0
    d.loc[(d['Fare']>7.91) & (d['Fare']<=14.54), 'Fare']=1
    d.loc[(d['Fare']>14.54) & (d['Fare']<=31), 'Fare']=2
    d.loc[(d['Fare']>31) & (d['Fare']<=513), 'Fare']=3
    d['Fare']=d['Fare'].astype(int)
train.head()

drop_col=['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Family']
train=train.drop(drop_col,axis=1)
test=test.drop(drop_col,axis=1)
test.head()
sns.heatmap(train.astype(float).corr(),annot=True,)
import xgboost as xgb
xgb1=xgb.XGBClassifier(n_estimators=2000,max_depth=4)
xgb1
p1={'gamma':[0.01,0.5,0.9],'colsample_bytree': [0.8,0.9,1], 'min_child_weight':[1,2],'max_depth': [3,4,5]}
g1 = GridSearchCV(xgb.XGBClassifier(),p1,refit=True,verbose=3)
rfc=RandomForestClassifier(n_estimators=1500,max_depth=5)
rfc
p2={'n_estimators': [1500,2000,2500],'min_weight_fraction_leaf': [0.01,0.05,0.1]}
g2 = GridSearchCV(RandomForestClassifier(),p2,refit=True,verbose=3)
xt=ExtraTreesClassifier(max_depth=4,n_estimators=2000)
xt
p3={'n_estimators': [1500,2000,2500],'min_weight_fraction_leaf': [0.01,0.05,0.1]}
g3 = GridSearchCV(ExtraTreesClassifier(),p3,refit=True,verbose=3)
svc=SVC()
svc
p4={'C': [0.01,0.1,1,10,100],'gamma': [0.0001,0.001,0.1,1,10],'kernel':['rbf']}
g4 = GridSearchCV(SVC(probability=True),p4,refit=True,verbose=3)
y_train=train['Survived'].ravel()
train=train.drop('Survived',axis=1)
X_train=train.values
X_test=test.values
from sklearn.ensemble import VotingClassifier
kfold = model_selection.KFold(n_splits=10)
# create the sub models
estimators = []
estimators.append(('xgboost', g1))
estimators.append(('Random forest', g2))
estimators.append(('extra tree', g3))
estimators.append(('svm', g4))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
print(results.mean())
g1.fit(X_train,y_train)
g2.fit(X_train,y_train)
g3.fit(X_train,y_train)
g4.fit(X_train,y_train)

def pred(X,g1,g2,g3,g4):
    y1=g1.predict_proba(X)[:,1]
    y2=g2.predict_proba(X)[:,1]
    y3=g3.predict_proba(X)[:,1]
    y4=g4.predict_proba(X)[:,1]
    return y1,y2,y3,y4
from sklearn.metrics import classification_report,confusion_matrix
def final_prob(y1,y2,y3,y4):
    y_fin=0.3*y1+0.25*y2+0.25*y3+0.2*y4
    return y_fin

def combine(y_fin):
    y_final=y_fin.copy()
    for i in range(0,len(y_fin)):
        if y_fin[i]>0.5:
            y_final[i]=1
        else:
            y_final[i]=0
    return y_final


ensemble.fit(X_train,y_train)
predictions=ensemble.predict(X_test)
predictions
s_xgb=g1.score(X_train,y_train)
print('xgb:', s_xgb)
s_rfc=g2.score(X_train,y_train)
print('rfc:' ,s_rfc )
s_ext=g3.score(X_train,y_train)
print('ext:', s_ext )
s_svc=g4.score(X_train,y_train)
print('svc:', s_svc )
s_ens=ensemble.score(X_train,y_train)
print('ensemble:', s_ens )

[y1_train,y2_train,y3_train,y4_train]=pred(X_test,g1,g2,g3,g4)
y_fin=final_prob(y1_train,y2_train,y3_train,y4_train)
y_final=combine(y_fin)
y_final=y_final.astype('int64')
y_final
predictions
new_test=pd.read_csv('../input/titanic/test (1).csv')
new_test.info()
submission = pd.DataFrame({"PassengerId": new_test["PassengerId"],"Survived": y_final})
submission.head()
submission.to_csv('submission.csv', index=False)
submission.info()

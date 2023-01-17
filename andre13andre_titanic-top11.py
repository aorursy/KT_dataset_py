# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas_profiling

import xgboost as xgb

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

test=pd.read_csv('/kaggle/input/titanic/test.csv',index_col='PassengerId')

train=pd.read_csv('/kaggle/input/titanic/train.csv',index_col='PassengerId')

train['dataset']='train'

test['dataset']='test'

test['Survived']= np.nan

dataset=train.append(test)
train.info()
dataset.describe()
train.boxplot()
train['Age'].plot.box()
profile=pandas_profiling.ProfileReport(train)

profile.to_file(output_file="report_titanic.html")
pd.plotting.scatter_matrix(train,figsize=(16,16))
sns.heatmap(dataset.isnull(), cbar = False).set_title("Missing values heatmap")
sns.distplot(train['Fare'])
sns.distplot(train['Age'])
sns.countplot(x = "SibSp", hue = "Survived", data = train)
sns.countplot(x = "Pclass", hue = "Survived", data = train)
sns.countplot(x = "Parch", hue = "Survived", data = train)
train['Fare'].describe()
dataset = dataset.fillna(np.nan)

dataset[['Last','F']] = dataset.Name.str.split(', ',expand=True)

dataset[['Pref','First']]=dataset.F.str.split('.',n=1,expand=True)

dataset.drop(['F','Name'],axis=1,inplace=True)



print('Train',train['Age'].min())

print('Test',test['Age'].min())

print('Train',train['Age'].max())

print('Test',test['Age'].max())

print('Train',train['Embarked'].unique())

print('Test',test['Embarked'].unique())

print('Train',train['Parch'].unique())

print('Test',test['Parch'].unique())

print('Train',train['Pclass'].unique())

print('Test',test['Pclass'].unique())

print('Train',train['SibSp'].unique())

print('Test',test['SibSp'].unique())



dataset.isnull().sum()
print((dataset[dataset['Pref']=='Master']['Age']).mean())

print((dataset[dataset['Pref']=='Master']['Age']).max())

print((dataset[dataset['Pref']=='Master']['Age']).min())

dataset['Age_isnull']=0

dataset.loc[dataset['Age'].isnull(),'Age_isnull']=1



#dataset.loc[(dataset['Age'].isnull()) & (dataset['Pref']=='Master'),'Age']=(dataset[dataset['Pref']=='Master']['Age']).mean()

#dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex']=='male'),'Age']=(dataset[(dataset['Sex']=='male') & (dataset['Pref']!='Master')]['Age']).mean()

#dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex']=='female'),'Age']=(dataset[(dataset['Sex']=='female') & (dataset['Pref']!='Master')]['Age']).mean()



dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())

dataset["Embarked"] = dataset["Embarked"].fillna("S")
grp = dataset.groupby(['Sex', 'Pclass'])  

dataset.Age = grp.Age.apply(lambda x: x.fillna(x.median()))



#If still any row remains

dataset.Age.fillna(dataset.Age.median, inplace = True)
dataset.head()
dataset['FamSize']=dataset['Parch']+dataset['SibSp']

dataset['Alone']=0

dataset.loc[dataset['FamSize']<1,'Alone']=1

#dataset['SmallFam']=0

#dataset['MidFam']=0

#dataset['BigFam']=0

#dataset.loc[dataset['FamSize']==1,'SmallFam']=1

#dataset.loc[(dataset['FamSize']==2) | (dataset['FamSize']==3),'MidFam']=1

#dataset.loc[dataset['FamSize']>3,'BigFam']=1

dataset['Fare'].describe()
plt.hist(dataset['Fare'],bins=5)
'''

dataset.loc[ dataset['Age'] <= 16, 'Ageclass'] = 0

dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Ageclass'] = 1

dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Ageclass'] = 2

dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Ageclass'] = 3

dataset.loc[ dataset['Age'] > 64, 'Ageclass']=4

that equal to 

dataset['Ageclass'] = pd.cut(dataset['Age'].astype(int), 5)

'''

dataset['Ageclass'] = pd.cut(dataset['Age'], [0, 10, 20, 30, 40, 50, 60,70,80])

dataset['FareClass'] = pd.cut(dataset['Fare'],[0,7.90,14.45,31.28,120], labels=['Low','Mid','High_Mid','High'])

sns.countplot(x = "FareClass", hue = "Survived", data = dataset)
dataset.head()
dataset['Cabin']=dataset['Cabin'].fillna('NA')



'''

encoding1=dataset.groupby('Embarked').size()

encoding1=encoding1/len(dataset)

dataset['Emb_freq']=dataset.Embarked.map(encoding1)

encoding2=dataset.groupby('Pclass').size()

encoding2=encoding2/len(dataset)

dataset['Pclass_freq']=dataset.Pclass.map(encoding2)





encoding4=dataset.groupby('FareClass').size()

encoding4=encoding4/len(dataset)

dataset['FareClass_freq']=dataset.FareClass.map(encoding4).astype(float)

encoding5=dataset.groupby('Ageclass').size()

encoding5=encoding5/len(dataset)

dataset['Ageclass_freq']=dataset.Ageclass.map(encoding5).astype(float)

'''
dataset.drop(['First','Last','Ticket','Fare','Age','SibSp','Parch'],axis=1,inplace=True)

le=LabelEncoder()

dataset['Sex']=le.fit_transform(dataset['Sex'])

dataset['Ageclass']=le.fit_transform(dataset['Ageclass'])

dataset=pd.get_dummies(dataset, columns = ['Ageclass','Cabin'],prefix=['Ageclass','Cabin'])

dataset=pd.get_dummies(dataset, columns = ['Embarked','Pref','Pclass','FareClass'],prefix=['Emb','Pref','Pclass','FareClass'],drop_first=True)
dataset.head()
train1=dataset[dataset['dataset']=='train'].drop(['dataset'],axis=1)

test1=dataset[dataset['dataset']=='test'].drop(['dataset','Survived'],axis=1)


#scaler = MinMaxScaler()

#train1[['Age','Fare','SibSp','Parch','FamSize']] = scaler.fit_transform(train1[['Age','Fare','SibSp','Parch','FamSize']].to_numpy())

#test1[['Age','Fare','SibSp','Parch','FamSize']] = scaler.transform(test1[['Age','Fare','SibSp','Parch','FamSize']].to_numpy())

X=train1.drop(['Survived'],axis=1)

y=train1['Survived']

X_test=test1
id_pass=pd.Series(X_test.index,name='PassengerId')
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

cv_folds=5

early_stopping_rounds=50



alg = xgb.XGBClassifier(learning_rate =0.1,

                    n_estimators=1000,

                    max_depth=5,

                    min_child_weight=5,

                    gamma=0,

                    subsample=0.5,

                    colsample_bytree=0.8,

                    objective= 'binary:logistic',

                    nthread=4,

                    scale_pos_weight=1,

                    seed=27)    



xgb_param = alg.get_xgb_params()

xgtrain = xgb.DMatrix(x_train, label=y_train)

cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

                    metrics=['error'], early_stopping_rounds=50)

print(cvresult)

alg.set_params(n_estimators=cvresult.shape[0])

    

alg.fit(x_train, y_train,eval_metric='error')

predictions = alg.predict(x_test)    

print(accuracy_score(y_test, predictions))

val_prediction = alg.predict(X_test)

pred_surv = pd.Series(val_prediction.astype(int), name="Survived")



results = pd.concat([id_pass,pred_surv],axis=1)



results.to_csv("xgboost_titanic.csv",index=False)
#from xgboost import plot_importance

#plot_importance(alg).nlarge()


clf = RandomForestClassifier(criterion='entropy', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

clf.fit(x_train,  np.ravel(y_train))

print("RF Accuracy: "+repr(round(clf.score(x_test, y_test) * 100, 2)) + "%")



result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')

print('The cross validated score for Random forest is:',round(result_rf.mean()*100,2))

y_pred = cross_val_predict(clf,x_train,y_train,cv=10)

sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Confusion_matrix for RF', y=1.05, size=15)



result = clf.predict(X_test)

submission = pd.DataFrame({'PassengerId':X_test.index,'Survived':result})

submission.Survived = submission.Survived.astype(int)

print(submission.shape)

filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
'''

param_test = {

 'max_depth':range(3,11),

 'min_child_weight':range(1,8),

 'gamma':[i/10.0 for i in range(0,5)],

 'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]



}

gsearch4 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=128, max_depth=5,

 min_child_weight=5, gamma=0, subsample=0.5, colsample_bytree=0.8,

 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 

 param_grid = param_test, scoring='accuracy',n_jobs=4,iid=False, cv=5)

gsearch4.fit(X,y)

gsearch4.best_params_, gsearch4.best_score_

'''
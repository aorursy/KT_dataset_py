import numpy as np 

import pandas as pd 

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from seaborn import heatmap

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import chi2

import pandas_profiling

from matplotlib import pyplot as plt

import seaborn as sns

import re

from sklearn.metrics import accuracy_score 

from sklearn.tree import export_graphviz

%matplotlib inline
traindf=pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

testdf=pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')



traindf.columns
cor=traindf.corr()

mask = np.triu(np.ones_like(cor, dtype=np.bool))

heatmap(cor.abs(),mask=mask,annot=True)
print('Train DF')

for col in traindf:

    

    print(col,traindf[col].isna().sum())

print('-'*30)

print('Test DF')

for col in testdf:

    

    print(col,testdf[col].isna().sum())

traindf["Age"] = traindf["Age"].fillna(traindf.groupby("Pclass")["Age"].transform("mean"))

testdf["Age"] = testdf["Age"].fillna(testdf.groupby("Pclass")["Age"].transform("mean"))


traindf['Alone'] = np.where((traindf['Parch']==0) & (traindf['SibSp']==0), 1, 0)

traindf['NoFamily'] = traindf['SibSp']+traindf['Parch']

traindf['Cabin_N']=traindf['Cabin'].str.extract('(\d+)')

traindf['Cabin_N']=pd.to_numeric(traindf['Cabin_N'])

traindf['Cabin_Ch']=traindf['Cabin'].str.extract('(\D+)')



pattern = ",\\s(.*?)\."

traindf['Title']=traindf['Name'].str.extract(pattern)
testdf['Alone'] = np.where((testdf['Parch']==0) & (testdf['SibSp']==0), 1, 0)

testdf['NoFamily'] = testdf['SibSp']+testdf['Parch']

testdf['Cabin_N']=testdf['Cabin'].str.extract('(\d+)')

testdf['Cabin_N']=pd.to_numeric(testdf['Cabin_N'])

testdf['Cabin_Ch']=testdf['Cabin'].str.extract('(\D+)')

testdf['Title']=testdf['Name'].str.extract(pattern)
palette =["C3","C2"]



f, axes = plt.subplots(3,3,figsize=(15,15))



#f.delaxes(axes[2, 2])

plt.figure()



sns.countplot(x="Pclass",data=traindf,hue='Survived',ax=axes[0,0],palette=palette)

sns.countplot(x="Sex",data=traindf,hue='Survived',ax=axes[0,1],palette=palette)

sns.countplot(x="SibSp",data=traindf,hue='Survived',ax=axes[0,2],palette=palette)

sns.countplot(x="Parch",data=traindf,hue='Survived',ax=axes[1,0],palette=palette)

sns.countplot(x="Embarked",data=traindf,hue='Survived',ax=axes[1,1],palette=palette)

sns.countplot(x="Alone",data=traindf,hue='Survived',ax=axes[1,2],palette=palette)

sns.countplot(x="NoFamily",data=traindf,hue='Survived',ax=axes[2,0],palette=palette)

sns.countplot(x="Cabin_Ch",data=traindf,hue='Survived',ax=axes[2,1],palette=palette)

plt_t=sns.countplot(x="Title",data=traindf,hue='Survived',ax=axes[2,2],palette=palette)

plt_t.set_xticklabels(plt_t.get_xticklabels(), rotation=90)

plt.show()
cont_tbl=pd.crosstab(traindf['Title'], traindf['Survived'])

cont_tbl_perc= pd.crosstab(traindf['Title'], traindf['Survived']).apply(lambda r: r/r.sum(), axis=1)

pd.concat([cont_tbl,cont_tbl_perc],axis=1)

NoFamily_Cat_dict={}

for key in [0,4,5,6]:

    NoFamily_Cat_dict[key]=1



for key in [1,2,3]:

    NoFamily_Cat_dict[key]=2    



    

for key in [7,10]:

    NoFamily_Cat_dict[key]=3

    



    

traindf['NoFamily_Cat']=traindf['NoFamily'].map(NoFamily_Cat_dict)

testdf['NoFamily_Cat']=testdf['NoFamily'].map(NoFamily_Cat_dict)

Parch_Cat_dict={}

for key in [0,5]:

    Parch_Cat_dict[key]=1



for key in [1,3]:

    Parch_Cat_dict[key]=2    



    

for key in [4,6]:

    Parch_Cat_dict[key]=3

    



for key in [2]:

    Parch_Cat_dict[key]=4

    

traindf['Parch_Cat']=traindf['Parch'].map(Parch_Cat_dict)

testdf['Parch_Cat']=testdf['Parch'].map(Parch_Cat_dict)

Cabin_Cat_dict={}

for key in ['C','E','D','B','F']:

    Cabin_Cat_dict[key]=1



for key in ['G']:

    Cabin_Cat_dict[key]=2    



    

for key in ['A']:

    Cabin_Cat_dict[key]=3

    



for key in ['F G','T']:

    Cabin_Cat_dict[key]=4





for key in ['F E']:

    Cabin_Cat_dict[key]=5



traindf['Cabin_Ch_Cat']=traindf['Cabin_Ch'].map(Cabin_Cat_dict)

testdf['Cabin_Ch_Cat']=testdf['Cabin_Ch'].map(Cabin_Cat_dict)
Title_Cat_dict={}

for key in ['Capt','Don','Jonkheer','Rev']:

    Title_Cat_dict[key]=1



for key in ['Col','Major','Dr','Master']:

    Title_Cat_dict[key]=2    



    

for key in ['Miss','Mrs']:

    Title_Cat_dict[key]=3

    



for key in ['Lady','Mlle','Mme','Ms','the Countess','Sir']:

    Title_Cat_dict[key]=4









traindf['Title_Cat']=traindf['Title'].map(Title_Cat_dict)

testdf['Title_Cat']=testdf['Title'].map(Title_Cat_dict)
continouos_cols=['Age','SibSp','Parch','Fare']

f, axes = plt.subplots(3,2,figsize=(15,15))

plt.figure()

f.delaxes(axes[2, 1])



sns.boxplot(x=traindf['Survived'],y=traindf['Age'],ax=axes[0,0],palette=palette)

sns.boxplot(x=traindf['Survived'],y=traindf['SibSp'],ax=axes[0,1],palette=palette)

sns.boxplot(x=traindf['Survived'],y=traindf['Parch'],ax=axes[1,0],palette=palette)

sns.boxplot(x=traindf['Survived'],y=traindf['Fare'],ax=axes[1,1],palette=palette)

sns.boxplot(x=traindf['Survived'],y=traindf['Cabin_N'],ax=axes[2,0],palette=palette)

plt.show()

sns.distplot(traindf.loc[traindf['Survived'] == 0]['Age'] ,color="r")

sns.distplot(traindf.loc[traindf['Survived'] == 1]['Age'],color="g")
sns.distplot(traindf.loc[traindf['Survived'] == 0]['Cabin_N'] ,color="r")

sns.distplot(traindf.loc[traindf['Survived'] == 1]['Cabin_N'],color="g")

traindf['Age_Quart']=pd.cut(traindf['Age'], bins=[0, 16, 30,60,100])

testdf['Age_Quart']=pd.cut(testdf['Age'], bins=[0, 16, 30,60,100])
traindf['Cabin_N_Quart']=pd.cut(traindf['Cabin_N'], bins=[0, 50, 100,150,1000])

testdf['Cabin_N_Quart']=pd.cut(testdf['Cabin_N'], bins=[0, 50, 100,150,1000])
traindf["Cabin_N"] = traindf["Cabin_N"].fillna(traindf.groupby("Pclass")["Cabin_N"].transform("mean"))

traindf["Cabin_Ch"] = traindf["Cabin_Ch"].fillna(traindf.mode().iloc[0]['Cabin_Ch'])

traindf["Embarked"] = traindf["Embarked"].fillna(traindf.mode().iloc[0]['Embarked'])
testdf["Cabin_N"] = testdf["Cabin_N"].fillna(testdf.groupby("Pclass")["Cabin_N"].transform("mean"))

testdf["Cabin_Ch"] = testdf["Cabin_Ch"].fillna(testdf.mode().iloc[0]['Cabin_Ch'])

testdf["Embarked"] = testdf["Embarked"].fillna(testdf.mode().iloc[0]['Embarked'])

testdf["Fare"]=testdf["Fare"].fillna(np.mean(testdf['Fare']))
traindf.drop(columns=['Ticket','Name','Cabin','Age','Cabin_Ch','NoFamily','Parch','Cabin_N','Title'],inplace=True)



traindf=pd.get_dummies(data=traindf, columns=['Sex','Cabin_Ch_Cat','Alone','Age_Quart','Embarked','NoFamily_Cat','Parch_Cat',

                                                'Cabin_N_Quart','Title_Cat'])

testdf.drop(columns=['Ticket','Name','Cabin','Age','Cabin_Ch','NoFamily','Parch','Cabin_N','Title'],inplace=True)



testdf=pd.get_dummies(data=testdf, columns=['Sex','Cabin_Ch_Cat','Alone','Age_Quart','Embarked','NoFamily_Cat','Parch_Cat',

                                                'Cabin_N_Quart','Title_Cat'])

print('Train DF')

for col in traindf:

    

    print(col,traindf[col].isna().sum())

print('-'*30)

print('Test DF')

for col in testdf:

    

    print(col,testdf[col].isna().sum())

y=traindf['Survived'].copy()

X=traindf.copy().drop('Survived',axis=1)

X_test=testdf.copy()



scaler=MinMaxScaler()



lc=LogisticRegression(penalty='l1', solver='liblinear')

pipe=Pipeline(steps=[('scaler',scaler),('lc',lc)])

grid_params={'lc__C':[0.01,0.1,1,10,100,1000]}

searchL=GridSearchCV(pipe,grid_params)

searchL.fit(X,y)



print("Best parameter (CV score=%0.3f):" % searchL.best_score_)

print(searchL.best_params_)
lc=LogisticRegression(penalty='l2',max_iter=5000)

pipe=Pipeline(steps=[('scaler',scaler),('lc',lc)])

grid_params={'lc__C':[0.01,0.1,1,10,100,1000]}

searchR=GridSearchCV(pipe,grid_params)

searchR.fit(X,y)



print("Best parameter (CV score=%0.3f):" % searchR.best_score_)

print(searchR.best_params_)
lc=RandomForestClassifier(random_state=14)

pipe=Pipeline(steps=[('scaler',scaler),('lc',lc)])

grid_params={'lc__n_estimators':[220,250,270],'lc__max_depth':[6,7,8]}

searchRF=GridSearchCV(pipe,grid_params)

searchRF.fit(X,y)



print("Best parameter (CV score=%0.3f):" % searchRF.best_score_)

print(searchRF.best_params_)
missing_cols = set( X.columns ) - set( X_test.columns )



for c in missing_cols:

    X_test[c] = 0



X_test = X_test[X.columns]
model=RandomForestClassifier(random_state=21,n_estimators=220,max_depth=7)



X=scaler.fit_transform(X)

X_test=scaler.transform(X_test)



model.fit(X,y)

pred_test=model.predict(X_test)

passengerId_test=testdf.index



pd.DataFrame({'Survived':pred_test,'PassengerId':passengerId_test}).to_csv('Submission_Final.csv',index=False)
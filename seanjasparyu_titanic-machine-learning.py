import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn import preprocessing

from sklearn.impute import SimpleImputer

from scipy import stats

from sklearn.model_selection import train_test_split



train_raw=pd.read_csv("/kaggle/input/titanic/train.csv")

y_train=train_raw['Survived']

test_raw=pd.read_csv("/kaggle/input/titanic/test.csv")

dfs=[train_raw, test_raw]

train_raw.head()
print('Train Data',train_raw.isnull().sum(),' ',sep='\n\n')

print('Test Data',test_raw.isnull().sum(),sep='\n\n')
dftrain_cor = train_raw[['Survived', 'Pclass', 'Age','SibSp','Parch','Fare']].copy()

train_cor=dftrain_cor.corr()



mask = np.zeros_like(train_cor)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(7, 5))

    ax = sns.heatmap(train_cor, mask=mask, annot=True,linewidths=1, vmax=.3, square=True)
plt.subplot2grid((1,2),(0,0))

sns.countplot(x='Survived',hue='Sex',data=train_raw)

plt.ylabel('Frequency')

plt.title("# of Survived")



plt.subplot2grid((1,2),(0,1))

sns.countplot(x='Embarked',hue='Survived',data=train_raw)

plt.ylabel(' ')

plt.title("# of Embarked")

plt.show()



plt.subplot2grid((1,2),(0,0))

sns.countplot(x='Pclass',hue='Sex',data=train_raw)

plt.ylabel('Frequency')

plt.title("# of Each Sex")



plt.subplot2grid((1,2),(0,1))

sns.countplot(x='Pclass',hue='Survived',data=train_raw)

plt.ylabel(' ')

plt.title("# of Survived")

plt.show()



fig, ax = plt.subplots()

sns.distplot(train_raw.Age[train_raw.Survived==1],kde=False,ax=ax, color="#1f77b4")

sns.distplot(train_raw.Age[train_raw.Survived==0],kde=False,ax=ax, color="#ff7f0e")

plt.title("Age Distribution")

plt.xlabel('Age')

red_patch = mpatches.Patch(color='#1f77b4', label='Survived')

blue_patch = mpatches.Patch(color='#ff7f0e', label='Died')

plt.legend(handles=[red_patch, blue_patch] ,loc='best')

plt.show()



fig, ax = plt.subplots()

sns.distplot(train_raw.Fare[train_raw.Pclass==1],kde=False,color="#1f77b4", ax=ax)

sns.distplot(train_raw.Fare[train_raw.Pclass==2],kde=False,color="#ff7f0e", ax=ax)

sns.distplot(train_raw.Fare[train_raw.Pclass==3],kde=False,color="#2ca02c", ax=ax)

plt.title("Fare Distribution")

patch_1 = mpatches.Patch(color='#1f77b4', label='Class 1')

patch_2 = mpatches.Patch(color='#ff7f0e', label='Class 2')

patch_3 = mpatches.Patch(color='#2ca02c', label='Class 3')

plt.legend(handles=[patch_1, patch_2, patch_3])

plt.xlabel('Fare')

plt.show()
#Pclass 1 Age Survival Figure

fig, ax = plt.subplots()

sns.distplot(train_raw.Age[train_raw.Survived==1][train_raw.Pclass==1],kde=False,ax=ax, color="#FFA500")

sns.distplot(train_raw.Age[train_raw.Survived==0][train_raw.Pclass==1],kde=False,ax=ax, color="#00FFFF")

red_patch = mpatches.Patch(color='#FFA500', label='Class 1 - Survived')

blue_patch = mpatches.Patch(color='#00FFFF', label='Class 1 - Died')

plt.legend(handles=[red_patch, blue_patch] ,loc='best')

#Pclass 2 Age Survival Figure

fig, ax = plt.subplots()

sns.distplot(train_raw.Age[train_raw.Survived==1][train_raw.Pclass==2],kde=False,ax=ax, color="#FFA500")

sns.distplot(train_raw.Age[train_raw.Survived==0][train_raw.Pclass==2],kde=False,ax=ax, color="#00FFFF")

red_patch = mpatches.Patch(color='#FFA500', label='Class 2 - Survived')

blue_patch = mpatches.Patch(color='#00FFFF', label='Class 2 - Died')

plt.legend(handles=[red_patch, blue_patch] ,loc='best')

#Pclass 3 Age Survival Figure

fig, ax = plt.subplots()

sns.distplot(train_raw.Age[train_raw.Survived==1][train_raw.Pclass==3],kde=False,ax=ax, color="#FFA500")

sns.distplot(train_raw.Age[train_raw.Survived==0][train_raw.Pclass==3],kde=False,ax=ax, color="#00FFFF")

red_patch = mpatches.Patch(color='#FFA500', label='Class 3 - Survived')

blue_patch = mpatches.Patch(color='#00FFFF', label='Class 3 - Died')

plt.legend(handles=[red_patch, blue_patch] ,loc='best')
# Dictionary with all the titles

TitleDict = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty", \

             "Don": "Royalty", "Sir" : "Royalty","Dr": "Royalty","Rev": "Royalty", \

             "Countess":"Royalty", "Mme": "Mrs", "Mlle": "Miss", "Ms": "Mrs","Mr" : "Mr", \

             "Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}



for df in dfs:

    df['Embarked']=df[['Embarked']].fillna(train_raw.mode()['Embarked'][0])

    df['Fare']=df[['Fare']].fillna(train_raw['Fare'].median())

    df['Title']=df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title']=df.Title.map(TitleDict)

    df['Fam_size']=df['SibSp']+df['Parch']+1

    df.loc[(df.Title=='Miss')&(df.Parch!=0)&(df.Fam_size>1),'Title']='Fchild'

    df['Alone']=df['Fam_size'].apply(lambda x:1 if x==1 else 0)



age_vals = dfs[0].groupby(['Pclass','Sex','Title'])['Age'].mean()

for df in dfs:

    vals=df[df["Age"].isnull()].index.values.astype(int).tolist()

    for val in vals:

        df.loc[val,'Age']=age_vals[df.loc[val,'Pclass'],df.loc[val,'Sex'],df.loc[val,'Title']]
train=dfs[0]

test=dfs[1]

train=train.drop(['PassengerId','Name','Cabin','Survived','Ticket'],axis=1)

test=test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)

test['Title']=test[['Title']].fillna('Mrs')
train = pd.get_dummies(train)

test= pd.get_dummies(test)
train['boxAge']=stats.boxcox(train['Age'])[0]

trainfare=np.array(train['Fare'])

trainfare[trainfare<=0]=0.01

train['boxFare']=stats.boxcox(trainfare)[0]



testfare=np.array(test['Fare'])

testfare[testfare<=0]=0.01



test['boxAge']=stats.boxcox(test['Age'], stats.boxcox(train['Age'])[1])

test['boxFare']=stats.boxcox(testfare, stats.boxcox(trainfare)[1])

train=train.drop(['Age','Fare'], axis=1)

test=test.drop(['Age','Fare'], axis=1)

train['Ageclass']=train['boxAge']*train['Pclass']

test['Ageclass']=test['boxAge']*train['Pclass']

train.head()
min_max_scaler = preprocessing.MinMaxScaler()

train[['boxAge','boxFare','Ageclass']] = min_max_scaler.fit_transform(train[['boxAge','boxFare','Ageclass']])

test[['boxAge','boxFare','Ageclass']] = min_max_scaler.fit_transform(test[['boxAge','boxFare','Ageclass']])
X_train, X_test, y_train, y_test = train_test_split(train, y_train, test_size=0.2, random_state=42)
LogReg=LogisticRegression()

LogReg.fit(X_train,y_train)

log_reg_s=LogReg.score(X_test,y_test)

print(log_reg_s)

model_comp=pd.DataFrame({'ML Model':'Log_Reg','Score':[log_reg_s]})

predict=LogReg.predict(test)

results=pd.DataFrame({'PassengerId':test_raw['PassengerId'],'Survived':pd.Series(predict)})

results.to_csv('resultslogreg.csv',index=False)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,max_depth=5)

model.fit(X_train, y_train)

Rand_For_S=model.score(X_test, y_test)

model_comp=model_comp.append({'ML Model':'Rand_For','Score':Rand_For_S},ignore_index=True)

print(Rand_For_S)
predict = model.predict(test)

results=pd.DataFrame({'PassengerId':test_raw['PassengerId'],'Survived':pd.Series(predict)})

results.to_csv('resultsrt.csv',index=False)
from xgboost import XGBClassifier

my_modelxg = XGBClassifier(n_estimators=1000, learning_rate=0.1, subsample=0.9, colsample_bytree = 0.9)

my_modelxg.fit(X_train, y_train, 

             early_stopping_rounds=5,

             eval_set=[(X_test, y_test)], 

             verbose=False)
predict_cv = my_modelxg.predict(X_test)





from sklearn.metrics import accuracy_score

cv_score=accuracy_score(y_test,predict_cv)

model_comp=model_comp.append({'ML Model':'XGB','Score':cv_score},ignore_index=True)

print ("CV Score: ",cv_score)
predict = my_modelxg.predict(test)

results=pd.DataFrame({'PassengerId':test_raw['PassengerId'],'Survived':pd.Series(predict)})

results.to_csv('resultsXG.csv',index=False)
print(model_comp.sort_values(by=['Score'],ascending=False))
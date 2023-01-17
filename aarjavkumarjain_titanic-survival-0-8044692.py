import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_df=pd.read_csv('../input/titanic/train.csv')

train_df.head()
train_df.describe()
train_df.shape
test_df=pd.read_csv('../input/titanic/test.csv')

test_df.head()
test_df.describe()
test_df.shape
train_df.columns
sns.countplot(train_df.Pclass).set_title("Count of Pclass")

sns.barplot(x='Pclass',y='Survived',data=train_df,ci=None).set_title('Pclass v/s Survived')
sns.countplot(train_df.Sex).set_title('Count of Sex')
sns.barplot(x='Sex',y='Survived',data=train_df,ci=None).set_title("Sex v/s Survived")
sns.distplot(train_df.Age)
ax=sns.distplot(train_df['Age'][train_df['Survived']==1],hist=False,label='Survived')

sns.distplot(train_df['Age'][train_df['Survived']==0],hist=False,label='Death')
sns.countplot(train_df.SibSp)
sns.barplot(x='SibSp',y='Survived',data=train_df,ci=None)
sns.countplot(train_df.Parch)
sns.barplot(x='Parch',y='Survived',data=train_df,ci=None)
sns.distplot(train_df.Fare)
ax=sns.distplot(train_df['Fare'][train_df['Survived']==1],label='Survived',hist=False)

sns.distplot(train_df['Fare'][train_df['Survived']==0],label='Death',hist=False,ax=ax)
sns.countplot(train_df.Embarked)
sns.barplot(x='Embarked',y='Survived',data=train_df,ci=None)
train_df['Title']=train_df['Name'].str.extract('([A-Za-z]+)\.')

train_df.head()
train_df.Title.value_counts()
title_dict={

    'Mr':'Mr',

    'Miss':'Miss',

    'Mrs':'Mrs',

    'Master':'Master',

    'Dr':'Other',

    'Rev':'Other',

    'Major':'Other',

    'Mlle':'Other',

    'Col':'Other',

    'Jonkheer':'Other',

    'Lady':'Other',

    'Sir':'Other',

    'Mme':'Other',

    'Don':'Other',

    'Capt':'Other',

    'Ms':'Other',

    'Countess':'Other',

}
train_df.Title=train_df.Title.map(title_dict)

train_df.Title.value_counts()
sns.countplot(train_df.Title)
sns.barplot(x='Title',y='Survived',data=train_df,ci=None)
train_df.groupby(['Title']).mean()
train_df.Age.fillna(train_df.groupby('Title')['Age'].transform('mean'),inplace=True)
bins=np.linspace(train_df.Age.min(),train_df.Age.max(),6)

group_name=['Children','Adult','Maturity','Aging','OldAge']

train_df['Age_binning']=pd.cut(train_df.Age,bins,labels=group_name,include_lowest=True)

train_df.head()
sns.countplot(train_df.Age_binning)
sns.barplot(x='Age_binning',y='Survived',data=train_df,ci=None)
bin_fare=np.linspace(train_df.Fare.min(),train_df.Fare.max(),4)

group_fare=['Low','Medium','High']

train_df['Fare_binning']=pd.cut(train_df.Fare,bin_fare,labels=group_fare,include_lowest=True)

train_df.head()
sns.countplot(train_df.Fare_binning)
sns.barplot(x='Fare_binning',y='Survived',data=train_df,ci=None)
train_df['Alone']=np.where((train_df["SibSp"]+train_df["Parch"])>0, 0, 1)
test_df['Title']=test_df['Name'].str.extract('([A-Za-z]+)\.')

test_df.head()
test_df.Title.value_counts()
title_dict_2={

    'Mr':'Mr',

    'Miss':'Miss',

    'Mrs':'Mrs',

    'Master':'Master',

    'Dr':'Other',

    'Rev':'Other',

    'Col':'Other',

    'Ms':'Other',

    'Dona':'Other'

}
test_df.Title=test_df.Title.map(title_dict_2)

test_df.Title.value_counts()
test_df.groupby(['Title']).mean()
test_df.Age.fillna(test_df.groupby('Title')['Age'].transform('mean'),inplace=True)
bins_test=np.linspace(test_df.Age.min(),test_df.Age.max(),6)

group_name_test=['Children','Adult','Maturity','Aging','OldAge']

test_df['Age_binning']=pd.cut(test_df.Age,bins_test,labels=group_name_test,include_lowest=True)

test_df.head()
test_df.Fare.fillna(train_df.groupby('Fare_binning')['Fare'].transform('mean'),inplace=True)
bin_fare_test=np.linspace(test_df.Fare.min(),test_df.Fare.max(),4)

group_fare_test=['Low','Medium','High']

test_df['Fare_binning']=pd.cut(test_df.Fare,bin_fare_test,labels=group_fare_test,include_lowest=True)

test_df.head()
test_df['Alone']=np.where((test_df["SibSp"]+test_df["Parch"])>0, 0, 1)
train_df.isna().sum()
print('Cabin Null Percentage: ',train_df.Cabin.isna().sum()/train_df.shape[0]*100)
train_df.drop('Cabin',axis=1,inplace=True)
train_df.Embarked.value_counts()
train_df.Embarked.fillna('S',inplace=True)
test_df.isna().sum()
test_df.drop('Cabin',axis=1,inplace=True)
train_df.head()
train_df.drop(['Name','Age','SibSp','Parch','Ticket','Fare'],axis=1,inplace=True)
test_df.drop(['Name','Age','SibSp','Parch','Ticket','Fare'],axis=1,inplace=True)
train_df.head()
dummies_train=pd.get_dummies(train_df[['Sex','Embarked','Title','Age_binning','Fare_binning']],drop_first=True)
dummies_test=pd.get_dummies(test_df[['Sex','Embarked','Title','Age_binning','Fare_binning']],drop_first=True)
train_df=pd.concat([train_df,dummies_train],axis=1)
test_df=pd.concat([test_df,dummies_test],axis=1)
train_df.drop(['Sex','Embarked','Title','Age_binning','Fare_binning'],axis=1,inplace=True)
test_df.drop(['Sex','Embarked','Title','Age_binning','Fare_binning'],axis=1,inplace=True)
train_df.head()
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import chi2
x_chi=train_df.drop(['PassengerId','Survived'],axis=1)

y_chi=train_df.Survived
features=[]

for col in x_chi.columns:

    res=chi2(train_df[[col]],y_chi)

    if res[1]<0.05:

        print(col,end=": ")

        print(res)

        features.append(col)
print(features)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
X=train_df[features]

y=train_df.Survived
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=23)
lm=LogisticRegression()
lm.fit(x_train,y_train)
yhat_lm=lm.predict(x_test)
f1_score_lm=f1_score(y_test,yhat_lm)

f1_score_lm
accuracy_score_lm=accuracy_score(y_test,yhat_lm)

accuracy_score_lm
tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
yhat_tree=tree.predict(x_test)
f1_score_tree=f1_score(y_test,yhat_tree)

f1_score_tree
accuracy_score_tree=accuracy_score(y_test,yhat_tree)

accuracy_score_tree
naive=GaussianNB()
naive.fit(x_train,y_train)
yhat_naive=naive.predict(x_test)
f1_score_naive=f1_score(y_test,yhat_naive)

f1_score_naive
accuracy_score_naive=accuracy_score(y_test,yhat_naive)

accuracy_score_naive
svc=SVC()
svc.fit(x_train,y_train)
yhat_svc=svc.predict(x_test)
f1_score_svc=f1_score(y_test,yhat_svc)

f1_score_svc
accuracy_score_svc=accuracy_score(y_test,yhat_svc)

accuracy_score_svc
neighbour=KNeighborsClassifier()
neighbour.fit(x_train,y_train)
yhat_neighbour=neighbour.predict(x_test)
f1_score_neighbour=f1_score(y_test,yhat_neighbour)

f1_score_neighbour
accuracy_score_neighbour=accuracy_score(y_test,yhat_neighbour)

accuracy_score_neighbour
forest=RandomForestClassifier()
forest.fit(x_train,y_train)
yhat_forest=forest.predict(x_test)
f1_score_forest=f1_score(y_test,yhat_forest)

f1_score_forest
accuracy_score_forest=accuracy_score(y_test,yhat_forest)

accuracy_score_forest
models_name=['LogisticRegression','DecisionTreeClassifier','GaussianNB','SVC','KNeighborsClassifier','RandomForestClassifier']
f1_score_models=[f1_score_lm,f1_score_tree,f1_score_naive,f1_score_svc,f1_score_neighbour,f1_score_forest]
fig,ax=plt.subplots(figsize=(10,6))

ax.bar(models_name,f1_score_models)

ax.set_title("F1 Score of  Test Data",pad=20)

ax.set_xlabel("Models",labelpad=20)

ax.set_ylabel("F1_Score",labelpad=20)

plt.xticks(rotation=90)



for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0%}'.format(height), (x+0.25, y + height + 0.01))
accuracy_score_models=[accuracy_score_lm,accuracy_score_tree,accuracy_score_naive,accuracy_score_svc,accuracy_score_neighbour,accuracy_score_forest]
fig,ax=plt.subplots(figsize=(10,6))

ax.bar(models_name,accuracy_score_models)

ax.set_title("Accuracy of Models on Test Data",pad=20)

ax.set_xlabel("Models",labelpad=20)

ax.set_ylabel("Accuracy",labelpad=20)

plt.xticks(rotation=90)



for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.0%}'.format(height), (x+0.25, y + height + 0.01))
forest.fit(train_df[features],train_df.Survived)
yhat_test_df=forest.predict(test_df[features])
test_df['Survived']=pd.Series(yhat_test_df)
submission_df=test_df[['PassengerId','Survived']]

submission_df.head()
submission_df.to_csv('answer_forest.csv',index=False,header=True)
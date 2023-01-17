#Impaorts

import pandas as pd

from pandas import Series,DataFrame

#numpy,matplotlib,seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



#machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
titanic_df=pd.read_csv("../input/train.csv")

test_df=pd.read_csv("../input/test.csv")

#print(titanic_df.head(3))

#print(test_df.head(3))

print(titanic_df.info())

print(test_df.info())

print(titanic_df.shape)

print(test_df.shape)
titanic_df=titanic_df.drop(['PassengerId','Name','Ticket'],axis=1)

test_df=test_df.drop(['Name','Ticket'],axis=1)

print(titanic_df.columns)

print(test_df.columns)
titanic_df["Embarked"]=titanic_df["Embarked"].fillna("S")

sns.factorplot('Embarked','Survived',data=titanic_df,size=4,aspect=3)

fig,(axis1,axis2,axis3)=plt.subplots(1,3,figsize=(10,5))

sns.countplot(x='Embarked',data=titanic_df,ax=axis1)

sns.countplot(x='Survived',hue="Embarked",data=titanic_df,order=[1,0],ax=axis2)

embark_perc=titanic_df[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean()

sns.barplot(x='Embarked',y='Survived',data=embark_perc,order=['S','C','Q'],ax=axis3)
embark_dummies_titanic=pd.get_dummies(titanic_df['Embarked'])

embark_dummies_titanic.drop(['S'],axis=1,inplace=True)



embark_dummies_test=pd.get_dummies(test_df['Embarked'])

embark_dummies_test.drop(['S'],axis=1,inplace=True)

titanic_df=titanic_df.join(embark_dummies_titanic)

test_df=test_df.join(embark_dummies_test)

titanic_df.drop(['Embarked'],axis=1,inplace=True)

test_df.drop(['Embarked'],axis=1,inplace=True)
print(test_df.loc[test_df["Fare"].isnull()])
test_df["Fare"].fillna(test_df["Fare"].median(),inplace=True)

titanic_df['Fare']=titanic_df['Fare'].astype(int)

test_df['Fare']=test_df['Fare'].astype(int)

fare_not_survived=titanic_df['Fare'][titanic_df['Survived']==0]

fare_survived=titanic_df['Fare'][titanic_df['Survived']==1]



avgerage_fare=DataFrame([fare_not_survived.mean()],[fare_survived.mean()])

std_fare=DataFrame([fare_not_survived.std()],[fare_survived.std()])



titanic_df['Fare'].plot(kind='hist',figsize=(15,3),bins=100,xlim=(0,50))

avgerage_fare.index.names=std_fare.index.names=["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,4))

axis1.set_title('Original Age values -Titanic')

axis2.set_title('New Age Values - Titanic')



average_age_titanic=titanic_df['Age'].mean()

std_age_titanic=titanic_df['Age'].std()

count_nan_age_titanic=titanic_df['Age'].isnull().sum()



average_age_test=test_df['Age'].mean()

std_age_test=test_df['Age'].std()

count_nan_age_test=test_df['Age'].isnull().sum()

rand_1=np.random.randint(average_age_titanic-std_age_titanic,average_age_titanic+std_age_titanic,size=count_nan_age_titanic)

rand_2=np.random.randint(average_age_test-std_age_test,average_age_test+std_age_test,size=count_nan_age_test)

titanic_df['Age'].dropna().astype(int).hist(bins=70,ax=axis1)

titanic_df['Age'][np.isnan(titanic_df['Age'])]=rand_1

test_df['Age'][np.isnan(test_df['Age'])]=rand_2



titanic_df['Age']=titanic_df['Age'].astype(int)

test_df['Age']=test_df['Age'].astype(int)

titanic_df['Age'].hist(bins=70,ax=axis2)
facet=sns.FacetGrid(titanic_df,hue='Survived',aspect=3)

facet.map(sns.kdeplot,'Age',shade=True)

facet.set(xlim=(0,titanic_df['Age'].max()))

facet.add_legend()



fig,axis1=plt.subplots(1,1,figsize=(12,4))

average_age=titanic_df[["Age","Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age',y='Survived',data=average_age)
titanic_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)
titanic_df['Family']=titanic_df['Parch']+titanic_df['SibSp']

titanic_df['Family'].loc[titanic_df['Family']>0]=1

titanic_df['Family'].loc[titanic_df['Family']==0]=0



test_df['Family']=test_df['Parch']+test_df['SibSp']

test_df['Family'].loc[test_df['Family']>0]=1

test_df['Family'].loc[test_df['Family']==0]=0

titanic_df=titanic_df.drop(['SibSp','Parch'],axis=1)

test_df=test_df.drop(['SibSp','Parch'],axis=1)



fig,(axis1,axis2)=plt.subplots(1,2,sharex=True,figsize=(10,5))

sns.countplot(x='Family',data=titanic_df,order=[1,0],ax=axis1)

family_perc=titanic_df[["Family","Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family',y='Survived',data=family_perc,order=[1,0],ax=axis2)

axis1.set_xticklabels(["With Family","Alone"],rotation=0)



print(titanic_df.columns)
def get_person(passenger):

    age,sex=passenger

    return 'child' if age<16 else sex



titanic_df['Person']=titanic_df[['Age','Sex']].apply(get_person,axis=1)

test_df['Person']=test_df[['Age','Sex']].apply(get_person,axis=1)



titanic_df.drop(['Sex'],axis=1,inplace=True)

test_df.drop(['Sex'],axis=1,inplace=True)



person_dummies_titanic=pd.get_dummies(titanic_df['Person'])

print(person_dummies_titanic.index.name)

person_dummies_titanic.columns=['Child','Female','Male']

person_dummies_titanic.drop(['Male'],axis=1,inplace=True)



person_dummies_test=pd.get_dummies(test_df['Person'])

person_dummies_test.columns=['Child','Female','Male']

person_dummies_test.drop(['Male'],axis=1,inplace=True)



titanic_df=titanic_df.join(person_dummies_titanic)

test_df=test_df.join(person_dummies_test)

print(titanic_df.columns)

print(test_df.columns)



fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,5))

sns.countplot(x='Person',data=titanic_df,ax=axis1)

person_perc=titanic_df[['Person','Survived']].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person',y='Survived',data=person_perc,ax=axis2,order=['male','female','child'])



titanic_df.drop(['Person'],axis=1,inplace=True)

test_df.drop(['Person'],axis=1,inplace=True)
sns.factorplot('Pclass','Survived',order=[1,2,3],data=titanic_df,size=5)
pclass_dummies_titanic=pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_titanic.columns=['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'],axis=1,inplace=True)



pclass_dummies_test=pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns=['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'],axis=1,inplace=True)



titanic_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



titanic_df=titanic_df.join(pclass_dummies_titanic)

test_df=test_df.join(pclass_dummies_test)
X_train=titanic_df.drop('Survived',axis=1)

Y_train=titanic_df['Survived']

X_test=test_df.drop(['PassengerId'],axis=1).copy()
logreg=LogisticRegression()

logreg.fit(X_train,Y_train)

Y_pred=logreg.predict(X_test)

print(logreg.score(X_train,Y_train))

print(logreg.score(X_test,Y_pred))
svc=SVC()

svc.fit(X_train,Y_train)

Y_pred=svc.predict(X_test)

svc.score(X_train,Y_train)
random_forest=RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train,Y_train)

Y_pred=random_forest.predict(X_test)

random_forest.score(X_train,Y_train)
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,Y_train)

Y_pred=knn.predict(X_test)

knn.score(X_train,Y_train)
gaussian=GaussianNB()

gaussian.fit(X_train,Y_train)

Y_pred=gaussian.predict(X_test)

gaussian.score(X_train,Y_train)
coeff_df=DataFrame(titanic_df.columns.delete(0))

print(coeff_df.info())

coeff_df.columns=['Features']

coeff_df["Coefficient Estimate"]=pd.Series(logreg.coef_[0])

print(coeff_df)
submission=pd.DataFrame({"PassengerId":test_df["PassengerId"],"Survived":Y_pred})

submission.to_csv('titanic.csv',index=False)
import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier

from sklearn.svm import SVC

from sklearn.cross_validation import KFold
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

PassengerId=test['PassengerId']
train.head(3)
test.head(3)
train['Fare'].value_counts()
full_data=[train,test]

train['Name_length']=train['Name'].apply(len)

test['Name_length']=test['Name'].apply(len)

train['Has_Cabin']=train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin']=test["Cabin"].apply(lambda x:0 if type(x) == float else 1)

for dataset in full_data:

    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1



for dataset in full_data:

    dataset['IsAlone']=0

    dataset.loc[dataset['FamilySize']==1,'IsAlone']=1

    

for dataset in full_data:

    dataset['Embarked']=dataset['Embarked'].fillna('S')

    

for dataset in full_data:

    dataset['Fare']=dataset['Fare'].fillna(train['Fare'].median())

    

train['CategoricalFare']=pd.qcut(train['Fare'].unique(),4)



for dataset in full_data:

    age_avg=dataset['Age'].mean()

    age_std=dataset['Age'].std()

    age_null_count=dataset['Age'].isnull().sum()

    age_null_random_list=np.random.randint(age_avg-age_std,age_avg+age_std,size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list

    dataset['Age']=dataset['Age'].astype(int)

    

train['CategoricalAge']=pd.cut(train['Age'],5)

def get_title(name):

    title_search=re.search(' ([A-Za-z]+\.)',name)

    if title_search:

        return title_search.group(1)

    return ""



for dataset in full_data:

    dataset['Title']=dataset['Name'].apply(get_title)

    

for dataset in full_data:

    dataset['Title']=dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')

    dataset['Title']=dataset['Title'].replace('Mlle','Miss')

    dataset['Title']=dataset['Title'].replace('Ms','Miss')

    dataset['Title']=dataset['Title'].replace('Mme','Mrs')



for dataset in full_data:

    dataset['Set']=dataset['Sex'].map({'female':0,'male':1}).astype(int)

    title_mapping={"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5}

    dataset['Title']=dataset['Title'].map(title_mapping)

    dataset['Title']=dataset['Title'].fillna(0)

    dataset['Embarked']=dataset['Embarked'].fillna('S')

    #dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )

    dataset.loc[dataset['Fare']<=7.91,'Fare']=0

    dataset.loc[(dataset['Fare']>7.91) &(dataset['Fare']<=14.454),'Fare']=1

    dataset.loc[(dataset['Fare']>14.454)&(dataset['Fare']<=31),'Fare']=2

    dataset.loc[(dataset['Fare']>31),'Fare']=3

    dataset['Fare']=dataset['Fare'].astype(int)

    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] 

print(dataset.columns)

print(train.columns)

#print(train[['Fare','CategoricalFare']])

print(train['Fare'].head(5))

print(train['CategoricalFare'].head(5))
print(train.shape)

print(type(train))

print(test.shape)

print(type(test))
print(type(train),train.columns,train.dtypes)

#print(train.info())

print(test.info())
test.drop(drop_elements,axis=1)

print('hello')
drop_elements=['PassengerId','Name','Ticket','Cabin','SibSp']

print(type(drop_elements))

train1=train.drop(drop_elements,axis=1)

train2=train1.drop(['CategoricalAge','CategoricalFare'],axis=1)

test1=test.drop(drop_elements,axis=1)
colormap=plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Person Correlation of Feature',y=1.05,size=15)

sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)

print("hello,world")
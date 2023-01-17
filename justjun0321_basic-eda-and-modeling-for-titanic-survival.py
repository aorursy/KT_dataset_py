import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from string import ascii_letters

import seaborn as sns



plt.style.use('ggplot')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
# Drop ID as it won't indicate too many things here

train.drop(['PassengerId'],axis=1,inplace=True)
pd.isnull(train).sum()
pd.isnull(test).sum()
train.describe()
test.describe()
for i in train.select_dtypes(include=['object']):

    print('Column ',i,' has ',train[i].nunique(),' unique values')
for i in test.select_dtypes(include=['object']):

    print('Column ',i,' has ',test[i].nunique(),' unique values')
target = train.Survived

train_df = train.drop('Survived',axis=1)

test_df = test.drop('PassengerId',axis=1)

train_df['is_train'] = 1

test_df['is_train'] = 0

train_test = pd.concat([train_df,test_df],axis=0)
train_test['Initial']=0

for i in train_test:

    train_test['Initial']=train_test.Name.str.extract('([A-Za-z]+)\.')

pd.crosstab(train_test.Initial,train_test.Sex).T.style.background_gradient(cmap='summer_r')
train['Initial']=0

for i in train:

    train['Initial']=train.Name.str.extract('([A-Za-z]+)\.')

pd.crosstab(train.Initial,train.Sex).T.style.background_gradient(cmap='summer_r')
pd.crosstab(train.Initial,train.Survived).T.style.background_gradient(cmap='summer_r')
train_test['Initial'].replace(['Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Master','Mlle','Mme','Ms','Rev','Sir'],

                            ['Special_male','Other_male','Special','Special_male','Special_female','Other','Special_male','Special','Special_male','Other_male','Special','Special','Special','Other_male','Special'],inplace=True)
train_test.groupby(['Initial','Pclass'])['Age'].aggregate(['mean','count'])
train_test.loc[(train_test.Age.isnull())]['Initial'].value_counts()
train_test.loc[(train_test.Age.isnull())]['Pclass'].value_counts()
train_test.loc[(train_test.Age.isnull())].groupby(['Initial','Pclass'])['Name'].aggregate('count')
from sklearn.preprocessing import Imputer



imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

temp_train_test = pd.get_dummies(train_test[['Initial','Pclass','Age']])

train_test_age_filled = imp.fit_transform(temp_train_test)
train_test_age_filled = pd.DataFrame(train_test_age_filled,columns=temp_train_test.columns.tolist())
train_test_age_filled.head()
train_test.Age = train_test_age_filled.Age
train_test.head()
train_test[np.isnan(train_test['Fare'])]
corr = train_test.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
train_test.groupby(['Pclass','Parch','SibSp'])['Fare'].mean()
train_test.loc[(train_test.Fare.isnull()),'Fare']=9
sns.countplot(train_test.Embarked)
train_test.loc[train_test.Embarked.isnull()]
train_test_temp = pd.get_dummies(train_test[['Fare','Parch','Pclass','Embarked']])



corr = train_test_temp.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
train_test.groupby(['Embarked','Pclass'])['Fare'].mean()
train_test["Embarked"] = train_test["Embarked"].fillna("S")
from sklearn.feature_selection import mutual_info_classif



train_temp = pd.get_dummies(train['Cabin'])



res = dict(zip(train_temp.columns.tolist(),

               mutual_info_classif(train_temp, train['Survived'], discrete_features=True)

               ))

print(res)
train_test.drop('Cabin',axis=1,inplace=True)
pd.isnull(train_test).sum()
train_df = train_test[train_test.is_train == 1]
train_df.drop(['is_train'],axis=1,inplace=True)
train_df['Survived'] = train['Survived']
train_df.head()
train_df.describe()
corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
f,ax=plt.subplots(2,3,figsize=(12,16))

sns.countplot('Pclass',data=train_df,ax=ax[0,0],hue='Survived')

ax[0,0].set_title('Pclass distribution')

sns.countplot('Sex',data=train_df,ax=ax[0,1],hue='Survived')

ax[0,1].set_title('Sex distribution')

sns.countplot('Initial',data=train_df,ax=ax[0,2],hue='Survived').set_xticklabels(rotation=40,labels=['Mr','Mrs','Miss','Other_male','Special_male','Other','Special'])

ax[0,2].set_title('Sex distribution')

sns.countplot('SibSp',data=train_df,ax=ax[1,0],hue='Survived')

ax[1,0].set_title('SibSp distribution')

sns.countplot('Parch',data=train_df,ax=ax[1,1],hue='Survived')

ax[1,1].set_title('Parch distribution')

sns.countplot('Embarked',data=train_df,ax=ax[1,2],hue='Survived')

ax[1,2].set_title('Embarked distribution')
pd.crosstab(train_df.Pclass,train_df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')
plt.title("Age & Survival Distribution")

plt.hist([train_df[train_df['Survived']==1]['Age'],train_df[train_df['Survived']==0]['Age']],bins = 10,label=['Survived', 'Dead'])

plt.legend()

plt.show()
train_df['Age_Range']=pd.qcut(train_df['Age'],7)

train_df.groupby(['Age_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
plt.title("Age & Survival Distribution")

plt.hist([train_df[train_df['Survived']==1]['Age'],train_df[train_df['Survived']==0]['Age']],bins = 25,label=['Survived', 'Dead'])

plt.legend()

plt.show()
sns.factorplot('Age_Range','Survived',hue='Sex',data=train_df)

plt.xticks(rotation=45)
plt.title("SibSp & Survival Distribution")

plt.hist([train_df[train_df['Survived']==1]['SibSp'],train_df[train_df['Survived']==0]['SibSp']],bins=10,range=[0,9],label=['Survived', 'Dead'])

plt.legend()

plt.show()
pd.crosstab(train_df.SibSp,train_df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')
plt.title("Parch & Survival Distribution")

plt.hist([train_df[train_df['Survived']==1]['Parch'],train_df[train_df['Survived']==0]['Parch']],bins=7,range=[0,7],label=['Survived', 'Dead'])

plt.legend()

plt.show()
pd.crosstab(train_df.Parch,train_df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')
plt.title("Fare & Survival Distribution")

plt.hist([train_df[train_df['Survived']==1]['Fare'],train_df[train_df['Survived']==0]['Fare']],bins=15,label=['Survived', 'Dead'])

plt.legend()

plt.show()
train_df['Fare_Range']=pd.qcut(train_df['Fare'],12)

train_df.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
train_df['Name_len'] = train_df.Name.apply(lambda x: len(x))
plt.title("Length of name & Survival Distribution")

plt.hist([train_df[train_df['Survived']==1]['Name_len'],train_df[train_df['Survived']==0]['Name_len']],bins=15,label=['Survived', 'Dead'])

plt.legend()

plt.show()
train_df['NameLen_Range']=pd.qcut(train_df['Name_len'],12)

train_df.groupby(['NameLen_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
sns.factorplot('NameLen_Range','Survived',hue='Pclass',data=train_df)

plt.xticks(rotation=45)
sns.countplot('NameLen_Range',hue='Survived',data=train_df)

plt.xticks(rotation=45)
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']
train_df.FamilySize.describe()
plt.title("FamilySize & Survival Distribution")

plt.hist([train_df[train_df['Survived']==1]['FamilySize'],train_df[train_df['Survived']==0]['FamilySize']],range(0,11),label=['Survived', 'Dead'])

plt.legend()

plt.show()
pd.crosstab(train_df.FamilySize,train_df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')
sns.countplot('Sex',hue='Survived',data=train_df)
pd.crosstab(train_df.Sex,train_df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')
sns.factorplot('Pclass','Survived',hue='Sex',data=train_df)
sns.factorplot('Fare_Range','Survived',hue='Pclass',data=train_df)

plt.xticks(rotation=45)
train_df.loc[(train_df['Pclass']==2) & (train_df['Fare'] == 0)]
train_test['Last_Name'] = train_test.Name.apply(lambda x: x.split(' (')[0].split('"')[0].split(' ')[-1])
train_test.head()
train_test['NameLen'] = train_test.Name.apply(lambda x : len(x))
train_test['Sex'] = train_test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train_test['FamilySize'] = train_test['Parch'] + train_test['SibSp']
train_test['Alone']=0

train_test.loc[(train_test.FamilySize==0),'Alone']=1
train_test['Small_Family']=0

train_test.loc[(train_test.FamilySize<=3)&(train_test.FamilySize!=0),'Small_Family']=1
train_test['Big_Family']=0

train_test.loc[(train_test.FamilySize>=4),'Big_Family']=1
train_test.head()
train_test.drop(['Name','SibSp','Parch','Ticket','FamilySize'],axis=1,inplace=True)
train_test = pd.get_dummies(train_test)
train_test.head()
train_test.info()
train = train_test[train_test.is_train == 1].drop(['is_train'],axis=1)



test = train_test[train_test.is_train == 0].drop(['is_train'],axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
X_train, X_test, Y_train, Y_test=train_test_split(train,train_df['Survived'],test_size=0.33,random_state=3)
model = LogisticRegression(random_state=51)

model.fit(X_train, Y_train)

prediction=model.predict(X_test)

print('Accuracy for rbf LogisticRegression is ',np.mean(cross_val_score(model, train, train_df['Survived'], cv=3)))
sns.heatmap(confusion_matrix(prediction,Y_test),annot=True,fmt='2.0f')
model = RandomForestClassifier(n_estimators=30,random_state=51)

model.fit(X_train, Y_train)

prediction=model.predict(X_test)

print('Accuracy for rbf RandomForestClassifier is ',np.mean(cross_val_score(model, train, train_df['Survived'], cv=3)))
sns.heatmap(confusion_matrix(prediction,Y_test),annot=True,fmt='2.0f')
model=svm.SVC(kernel='linear',C=0.1,gamma=0.1,random_state=51)

model.fit(X_train,Y_train)

prediction=model.predict(X_test)

print('Accuracy for linear SVM is',np.mean(cross_val_score(model, train, train_df['Survived'], cv=3)))
sns.heatmap(confusion_matrix(prediction,Y_test),annot=True,fmt='2.0f')
model=LinearSVC(random_state=51)

model.fit(X_train, Y_train)

prediction=model.predict(X_test)

print('The accuracy of the NaiveBayes is',np.mean(cross_val_score(model, train, train_df['Survived'], cv=3)))
sns.heatmap(confusion_matrix(prediction,Y_test),annot=True,fmt='2.0f')
model=RandomForestClassifier(n_estimators=30)

model.fit(X_train, Y_train)

prediction = model.predict(test)

submission = pd.read_csv('../input/gender_submission.csv')

submission.Survived = prediction

submission.to_csv('submission.csv',index=False)
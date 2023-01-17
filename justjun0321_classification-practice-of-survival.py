import pandas as pd

from pandas import Series,DataFrame

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

train_df.describe(include="all")
test_df.describe(include="all")
from string import ascii_letters

import seaborn as sns

corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
print(pd.isnull(train_df).sum())
train_df.head()
train_df = train_df.drop(['PassengerId'], axis=1)
print(pd.isnull(train_df).sum())
print(pd.isnull(test_df).sum())
train_df["Embarked"] = train_df["Embarked"].fillna("S")
#embark_dummies_train  = pd.get_dummies(train_df['Embarked'])
#embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
#train_df = train_df.join(embark_dummies_train)

#test_df = test_df.join(embark_dummies_test)



#train_df.drop(['Embarked'], axis=1,inplace=True)

#test_df.drop(['Embarked'], axis=1,inplace=True)
print(pd.isnull(train_df).sum())
print(pd.isnull(test_df).sum())
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare'] = test_df['Fare'].astype(int)
train_df['Initial']=0

for i in train_df:

    train_df['Initial']=train_df.Name.str.extract('([A-Za-z]+)\.')

pd.crosstab(train_df.Initial,train_df.Sex).T.style.background_gradient(cmap='summer_r')
train_df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
train_df.groupby('Initial')['Age'].mean()
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mr'),'Age']=33

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mrs'),'Age']=36

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Master'),'Age']=5

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Miss'),'Age']=22

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Other'),'Age']=46
plt.hist(train_df.Age)
train_df['Age_band']=0

train_df.loc[train_df['Age']<=16,'Age_band']=0

train_df.loc[(train_df['Age']>16)&(train_df['Age']<=24),'Age_band']=1

train_df.loc[(train_df['Age']>24)&(train_df['Age']<=32),'Age_band']=2

train_df.loc[(train_df['Age']>32)&(train_df['Age']<=48),'Age_band']=3

train_df.loc[(train_df['Age']>48)&(train_df['Age']<=64),'Age_band']=4

train_df.loc[train_df['Age']>64,'Age_band']=5
test_df['Initial']=0

for i in test_df:

    test_df['Initial']=test_df.Name.str.extract('([A-Za-z]+)\.')

pd.crosstab(test_df.Initial,test_df.Sex).T.style.background_gradient(cmap='summer_r')
test_df['Initial'].replace(['Ms','Dr','Col','Rev','Sir','Dona'],['Miss','Mr','Other','Other','Mr','Other'],inplace=True)
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mr'),'Age']=33

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mrs'),'Age']=36

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Master'),'Age']=5

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Miss'),'Age']=22

train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Other'),'Age']=46
plt.hist(test_df.Age,range=(test_df.Age.min(),test_df.Age.max()))
test_df['Age_band']=0

test_df.loc[test_df['Age']<=16,'Age_band']=0

test_df.loc[(test_df['Age']>16)&(test_df['Age']<=24),'Age_band']=1

test_df.loc[(test_df['Age']>24)&(test_df['Age']<=32),'Age_band']=2

test_df.loc[(test_df['Age']>32)&(test_df['Age']<=48),'Age_band']=3

test_df.loc[(test_df['Age']>48)&(test_df['Age']<=64),'Age_band']=4

test_df.loc[test_df['Age']>64,'Age_band']=5
corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
train_df['Family_Size']=0

train_df['Family_Size']=train_df['Parch']+train_df['SibSp']
test_df['Family_Size']=0

test_df['Family_Size']=test_df['Parch']+test_df['SibSp']
train_df['Alone']=0

test_df['Alone']=0

train_df.loc[train_df.Family_Size==0,'Alone']=1

test_df.loc[test_df.Family_Size==0,'Alone']=1
corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
train_df['Fare_Range']=pd.qcut(train_df['Fare'],4)

train_df.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')
train_df['Fare_new']=0

train_df.loc[train_df['Fare']<=7,'Fare_new']=0

train_df.loc[(train_df['Fare']>7)&(train_df['Fare']<=14),'Fare_new']=1

train_df.loc[(train_df['Fare']>14)&(train_df['Fare']<=31),'Fare_new']=2

train_df.loc[(train_df['Fare']>31)&(train_df['Fare']<=512),'Fare_new']=3



test_df['Fare_new']=0

test_df.loc[test_df['Fare']<=7,'Fare_new']=0

test_df.loc[(test_df['Fare']>7)&(test_df['Fare']<=14),'Fare_new']=1

test_df.loc[(test_df['Fare']>14)&(test_df['Fare']<=31),'Fare_new']=2

test_df.loc[(test_df['Fare']>31)&(test_df['Fare']<=512),'Fare_new']=3
mode = lambda x: x.mode() if len(x) > 2 else np.array(x)

train_df.groupby('Initial')['Cabin'].agg(mode)
train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Mr'),'Cabin']='B51'

train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Mrs'),'Cabin']='D'

train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Master'),'Cabin']='F2'

train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Miss'),'Cabin']='E101'

train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Other'),'Cabin']='A26'
test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Mr'),'Cabin']='B51'

test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Mrs'),'Cabin']='D'

test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Master'),'Cabin']='F2'

test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Miss'),'Cabin']='E101'

test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Other'),'Cabin']='A26'
train_df = train_df.drop(['Age'], axis=1)

test_df    = test_df.drop(['Age'], axis=1)
corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
train_df['Embarked'] = train_df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

test_df['Embarked'] = test_df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)
train_df = pd.get_dummies(train_df, columns=['Embarked','Initial','Parch','SibSp','Pclass'] )

test_df = pd.get_dummies(test_df, columns=['Embarked','Initial','Parch','SibSp','Pclass'] )
#pclass_dummies_train = pd.get_dummies(train_df['Pclass'])

#pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']



#pclass_dummies_test = pd.get_dummies(test_df['Pclass'])

#pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
#train_df.drop(['Pclass'],axis=1,inplace=True)

#test_df.drop(['Pclass'],axis=1,inplace=True)

#train_df = train_df.join(pclass_dummies_train)

#test_df = test_df.join(pclass_dummies_test)
train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
#pclass_dummies_train = pd.get_dummies(train_df['Sex'])

#pclass_dummies_train.columns = ['Sex1','Sex2']

#pclass_dummies_test = pd.get_dummies(test_df['Sex'])

#pclass_dummies_test.columns = ['Sex1','Sex2']

#train_df.drop(['Sex'],axis=1,inplace=True)

#test_df.drop(['Sex'],axis=1,inplace=True)

#train_df = train_df.join(pclass_dummies_train)

#test_df = test_df.join(pclass_dummies_test)
corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
#train_df = train_df.drop(['Sex1'], axis=1)

#test_df = test_df.drop(['Sex1'], axis=1)
train_df.info()

print("----------------------------")

test_df.info()
train_df = train_df.drop(['Name','Ticket','Fare_Range','Cabin'], axis=1)

test_df = test_df.drop(['Name','Ticket','Cabin'], axis=1)
corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})
#train_df = train_df.drop(['Fare','S','SibSp','Parch'], axis=1)

#test_df = test_df.drop(['Fare','S','SibSp','Parch'], axis=1)
X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
X_train.info()

print("----------------------------")

X_test.info()
test_df = test_df.drop(['Parch_9'], axis=1)
X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred1 = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
K_Neighbors = KNeighborsClassifier()



K_Neighbors.fit(X_train, Y_train)



Y_pred2 = K_Neighbors.predict(X_test)



K_Neighbors.score(X_train, Y_train)
from sklearn import metrics

model=GaussianNB()

model.fit(X_train,Y_train)

Y_pred3=model.predict(X_test)

model.score(X_train, Y_train)
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

model.fit(X_train,Y_train)

Y_pred4=model.predict(X_test)

model.score(X_train, Y_train)
def save_results(predictions, filename):

    with open(filename, 'w') as f:

        f.write("PassengerId,Survived\n")

        for i, pred in enumerate(predictions):

            f.write("%d,%f\n" % (i + 1, pred))
save_results(Y_pred,'1.csv' )

save_results(Y_pred1,'2.csv' )

save_results(Y_pred2,'3.csv' )

save_results(Y_pred3,'4.csv' )

save_results(Y_pred4,'5.csv' )
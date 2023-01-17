import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from pandas import Series,DataFrame



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.info()

print("---------------------------")

test.info()
train.describe()
#丢弃无用feature

train = train.drop(['PassengerId','Name','Ticket'],axis=1)

test = test.drop(['Name','Ticket'],axis=1)
sns.factorplot('Embarked',data=train,kind='count',order=['S','C','Q'])
#Embarked 有2个缺失值，因为'S'值最多，此处以'S'值填充

train['Embarked'] = train['Embarked'].fillna('S')
#plot

#分析Embarked与Survived的关系

sns.factorplot('Embarked','Survived', data=train,size=4,aspect=3)



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



#分析Embarked

sns.countplot(x='Embarked', data=train, ax=axis1)

#分析Embarked与Survived的关系

sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2)



#分析Embarked的取值与Survived的关系，算均值，看出‘Q’时，Survive概率最大

embark_perc = train[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
#分析后，对Embarked进行处理

#one-hot编码

embark_dummies_titanic  = pd.get_dummies(train['Embarked'])

#embark_dummies_titanic.head()



#'S'的属性对于结果没有太大帮助所以舍弃'S',或者丢弃整个Embarked特征，

#因为结果分析来看此特征对于Survive没有太大影响

embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

embark_dummies_titanic.head()



#处理test数据

embark_dummies_test  = pd.get_dummies(test['Embarked'])

embark_dummies_test.drop(['S'], axis=1, inplace=True)



train = train.join(embark_dummies_titanic)

test = test.join(embark_dummies_test)



train.drop(['Embarked'], axis=1,inplace=True)

test.drop(['Embarked'], axis=1,inplace=True)



train.head()
#对于test的fare属性有部分缺失值，用中值补全

test['Fare'].fillna(test['Fare'].median(),inplace=True)



#float to int ？？

train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].astype(int)





# 获得Survive与否下的passenger数据

fare_not_survived = train["Fare"][train["Survived"] == 0]

fare_survived     = train["Fare"][train["Survived"] == 1]



#分别计算均值和标准差

avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])



#Fare数据

train['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))

avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))

axis1.set_title('Original Age values - Titanic')

axis2.set_title('New Age values - Titanic')

print("orgin")

print(train['Age'].describe())



#获得train各个指标

average_age_titanic   = train["Age"].mean()

std_age_titanic       = train["Age"].std()

count_nan_age_titanic = train["Age"].isnull().sum()



#获得test各个指标

average_age_test   = test["Age"].mean()

std_age_test       = test["Age"].std()

count_nan_age_test = test["Age"].isnull().sum()



# 随机生成位于 (mean - std) & (mean + std)之间的数



rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)



#plot时需要删除所有Nan的数据

train['Age'].dropna().astype(int).hist(bins=70,ax=axis1)



# 用均值和标准差之间的随机数来填充缺失值

train["Age"][np.isnan(train["Age"])] = rand_1

test["Age"][np.isnan(test["Age"])] = rand_2



#转化成整数

train['Age']   = train['Age'].astype(int)

test['Age']    = test['Age'].astype(int)



# plot 新的Age数据

train['Age'].hist(bins=70, ax=axis2)

print('------------new value--------------')

train['Age'].describe()
'''

facet = sns.FacetGrid(train, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, train['Age'].max()))

facet.add_legend()

'''







fig, axis1 = plt.subplots(1,1,figsize=(18,4))

average_age = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data=average_age)
#缺失过多的值，直接剔除掉

train['Cabin'].describe()



train.drop("Cabin",axis=1,inplace=True)

test.drop("Cabin",axis=1,inplace=True)
#将SibSp 和 Parch合并成一个feature

#有亲属则设置为1，没有亲属则设置为0

train['Family'] =  train["Parch"] + train["SibSp"]

train['Family'].loc[train['Family'] > 0] = 1

train['Family'].loc[train['Family'] == 0] = 0



test['Family'] =  test["Parch"] + test["SibSp"]

test['Family'].loc[test['Family'] > 0] = 1

test['Family'].loc[test['Family'] == 0] = 0



# drop Parch & SibSp

train = train.drop(['SibSp','Parch'], axis=1)

test    = test.drop(['SibSp','Parch'], axis=1)



#画图分析

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



#family

sns.countplot(x='Family', data=train, order=[1,0], ax=axis1)



#family均值与Survive关系

family_perc = train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



axis1.set_xticklabels(["With Family","Alone"], rotation=0)
#通过分析，儿童(Age<16) Survival的概率很大，因此sex分为males，females，child

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex



#增加person字段来代替sex字段

train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)

test['Person'] = test[['Age','Sex']].apply(get_person,axis=1)



#删除Sex字段

train.drop(['Sex'],axis=1,inplace=True)

test.drop(['Sex'],axis=1,inplace=True)



#将Person字段one-hot编码，鉴于Male对于Survival帮助不大，因此舍去

person_dummies_titanic  = pd.get_dummies(train['Person'])

person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test['Person'])

person_dummies_test.columns = ['Child','Female','Male']

person_dummies_test.drop(['Male'], axis=1, inplace=True)



train = train.join(person_dummies_titanic)

test = test.join(person_dummies_test)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



#分析Person数据

sns.countplot(x='Person',data=train,ax=axis1)



person_perc  = train[['Person','Survived']].groupby(['Person'],as_index=False).mean()

sns.barplot(x='Person',y='Survived',data=person_perc,ax=axis2,order=['male','female','child'])



train.drop(['Person'],axis=1,inplace=True)

test.drop(['Person'],axis=1,inplace=True)
train['Pclass'].describe()

train['Pclass'].head()



sns.factorplot('Pclass','Survived', data=train,size=5)





#class=3 帮助很小，舍去，对Pclass进行one-hot编码

pclass_dummies_titanic  = pd.get_dummies(train['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



train.drop(['Pclass'],axis=1,inplace=True)

test.drop(['Pclass'],axis=1,inplace=True)



train = train.join(pclass_dummies_titanic)

test    = test.join(pclass_dummies_test)
X_train = train.drop('Survived',axis=1)

Y_train = train['Survived']

X_test = test.drop(['PassengerId'],axis=1).copy()
logreg = LogisticRegression()



logreg.fit(X_train, Y_train)



Y_logreg_pred = logreg.predict(X_test)



logreg.score(X_train, Y_train)
svc = SVC()



svc.fit(X_train, Y_train)



Y_svc_pred = svc.predict(X_test)



svc.score(X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_random_forest_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors = 3)



knn.fit(X_train, Y_train)



Y_knn_pred = knn.predict(X_test)



knn.score(X_train, Y_train)
gaussian = GaussianNB()



gaussian.fit(X_train, Y_train)



Y_gaussian_pred = gaussian.predict(X_test)



gaussian.score(X_train, Y_train)
coeff_df = DataFrame(train.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



coeff_df
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_random_forest_pred

    })

submission.to_csv('titanic.csv', index=False)
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
import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('ggplot')

sns.set(font_scale=2.5)



import missingno as msno



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
sns.set_style('whitegrid')

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.info()

df_train.shape
df_train.describe()
df_train.isnull().sum()
df_train.head()
for col in df_train.columns:

    percent = 'colum: {:>10}\t Percent of NA value {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(percent)
for col in df_test.columns:

    percent = 'colum: {:>10}\t Percent of NA value {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(percent)
msno.matrix(df=df_train.iloc[:,:], figsize=(8,8), color=(1,.5,.1))
msno.bar(df=df_train.iloc[:,:], figsize=(8,8), color=(1,.5,.1))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,8))



df_train["Survived"].value_counts().plot.pie(explode=[0,0.1], autopct = '%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel("")

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')
df_train["Survived"].value_counts()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
(df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).count())
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived')
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(figsize=(10,5))
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,8))



df_train["Pclass"].value_counts().plot.bar(ax=ax[0])

ax[0].set_title("# of passengers By Pclass")



sns.countplot('Pclass', hue="Survived", data=df_train,ax=ax[1])

ax[1].set_title("Pclass: Survived vs Dead")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,8))



df_train[["Sex","Survived"]].groupby(["Sex"], as_index=True).mean().plot.bar(ax=ax[0])



ax[0].set_title("Survived vs Sex")

sns.countplot("Sex", hue="Survived", data=df_train, ax=ax[1])

ax[1].set_title("Sex: Survived vs Dead")
df_train[["Sex","Survived"]].groupby(["Sex"], as_index=True).mean()
pd.crosstab(df_train["Sex"], df_train["Survived"], margins=True)
sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=df_train, size=6)
sns.factorplot(x="Sex", y='Survived', col='Pclass', data=df_train, size=6)
sns.factorplot(x="Sex", y='Survived', hue='Pclass', data=df_train, size=8)
print("the oldest one : {:.1f} years".format(df_train['Age'].max()))

print("the youngest one : {:.1f} years".format(df_train['Age'].min()))

print("average : {:.1f} years".format(df_train['Age'].mean()))
df_train["Age"].describe()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,8))



(df_train[df_train["Survived"]==0]['Age']).hist(ax=ax[0])

ax[0].set_title("Dead people")

(df_train[df_train["Survived"]==1]['Age']).hist(ax=ax[1])

ax[1].set_title("Survived people")
plt.figure(figsize=(8,6))



df_train[df_train["Pclass"] == 1]["Age"].plot(kind='kde')

df_train[df_train["Pclass"] == 2]["Age"].plot(kind='kde')

df_train[df_train["Pclass"] == 3]["Age"].plot(kind='kde')



plt.xlabel("Age")

plt.title("Age distribution within classes")

plt.legend(['1st Clas', '2nd Class', '3rd Class'])
df_train[df_train["Pclass"] == 1]["Age"].head()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))



sns.kdeplot(df_train[(df_train["Survived"]==0) & (df_train["Pclass"] == 1)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train["Survived"]==1) & (df_train["Pclass"] == 1)]['Age'], ax=ax)

plt.legend(["Survived == 0", 'Survived == 1'])
change_age_range_survival_ratio = list()



for num in range(1,80):

    change_age_range_survival_ratio.append(df_train[df_train["Age"] < num]['Survived'].sum()/ df_train[df_train['Age'] < num]['Survived'].shape[0])
plt.figure(figsize=(8,8))



plt.plot(change_age_range_survival_ratio)

# y=1.02를 쓰면 타이틀 위치가 괜찮아 보임.

plt.title("Survival rate chage depending on range of Age", y=1.02)

plt.ylabel("Survival rate")

plt.xlabel("Range of Age (0~x)")
change_age_range_survival_ratio[:5]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))



sns.violinplot('Pclass', "Age", hue='Survived', data=df_train, scale='count', split=True, ax=ax[0])

ax[0].set_title("Pclass and Age vs Survived")

ax[0].set_yticks(range(0,110,10))



sns.violinplot('Sex', "Age", hue='Survived', data=df_train, scale='count', split=True, ax=ax[1])

ax[1].set_title("Sex and Age vs Survived")

ax[1].set_yticks(range(0,110,10))

df_train[["Embarked", "Survived"]].groupby(["Embarked"], as_index=True).head()
f, ax = plt.subplots(1,1, figsize=(7,7))



df_train[["Embarked", "Survived"]].groupby(["Embarked"], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,20))



sns.countplot("Embarked", data=df_train, ax=ax[0,0])

ax[0,0].set_title('1. # of Passengers Board')



sns.countplot("Embarked", data=df_train, ax=ax[0,1], hue='Sex')

ax[0,1].set_title('2. # of Male/Female split for embarked')



sns.countplot("Embarked", data=df_train, ax=ax[1,0], hue='Survived')

ax[1,0].set_title('3. # of Embarked vs Survived')



sns.countplot("Embarked", data=df_train, ax=ax[1,1], hue='Pclass')

ax[1,1].set_title('4. # of Embarked vs Pclass')



plt.subplots_adjust(wspace = 0.5 , hspace=0.3)
df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"]

df_train.describe()
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(40,10))



sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('1 # of Passenger Board', y=1.02)



sns.countplot('FamilySize', hue='Survived',data=df_train, ax=ax[1])

ax[1].set_title('2 # of Survived countplot depending on FamilySize', y=1.02)



df_train[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index=True).mean().sort_values(by="Survived", ascending=False).head().plot.bar(ax=ax[2])

ax[2].set_title('3 # of Passenger Board', y=1.02)



plt.subplots_adjust(wspace=.2, hspace=.2)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

pl = sns.distplot(df_train["Fare"], label="Skewness: {:.2f}".format(df_train["Fare"].skew()), ax=ax)

plots = pl.legend(loc='best')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

pl = sns.distplot(np.log(df_train["Fare"] + .1), label="Skewness: {:.2f}".format(df_train["Fare"].skew()), ax=ax)

plots = pl.legend(loc='best')
df_train["Fare.log"] = df_train["Fare"].map(lambda i : np.log(i) if i > 0 else 0)

df_train.head()
df_train[(df_train["Fare"] == 0)].head()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))

pl = sns.distplot(df_train["Fare.log"], label="Skewness: {:.2f}".format(df_train["Fare.log"].skew()), ax=ax)

plots = pl.legend(loc='best')
df_train['Ticket'].value_counts().head(10)
df_train["Age"].isnull().sum()
df_train["Name"].head()
df_train["Name"].str.extract('([A-Za-z]+)\.')[:10]
df_train["Initial"] = df_train["Name"].str.extract('([A-Za-z]+)\.')

df_test["Initial"] = df_test["Name"].str.extract('([A-Za-z]+)\.')
pd.crosstab(df_train['Initial'], df_train["Sex"]).T
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
pd.crosstab(df_train['Initial'], df_train["Sex"]).T
df_train.groupby("Initial").mean()
df_train.groupby("Initial")["Survived"].mean().plot.bar()
df_train.loc[(df_train["Age"].isnull()), :].head()
df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] =="Mr"), :].head()
df_train.groupby("Initial").mean()
df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Mr"), "Age"] = 33

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Mrs"), "Age"] = 36

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Master"), "Age"] = 5

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Miss"), "Age"] = 22

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Other"),"Age"] = 46
df_train["Age"].isnull().sum()
df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Mr"), "Age"] = 33

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Mrs"), "Age"] = 36

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Master"), "Age"] = 5

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Miss"), "Age"] = 22

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Other"), "Age"] = 46
df_test["Age"].isnull().sum()
df_train["Embarked"].isnull().sum()
df_train.shape
df_train["Embarked"].fillna('S', inplace=True)
df_train["Embarked"].isnull().sum()
df_train.head()
df_train['Age_cat'] = 0

df_train.head()
df_train.loc[df_train["Age"] < 10, 'Age_cat'] = 0 

df_train.loc[(10 <= df_train["Age"]) & (df_train["Age"] < 20), 'Age_cat'] = 1

df_train.loc[(20 <= df_train["Age"]) & (df_train["Age"] < 30), 'Age_cat'] = 2

df_train.loc[(30 <= df_train["Age"]) & (df_train["Age"] < 40), 'Age_cat'] = 3

df_train.loc[(40 <= df_train["Age"]) & (df_train["Age"] < 50), 'Age_cat'] = 4

df_train.loc[(50 <= df_train["Age"]) & (df_train["Age"] < 60), 'Age_cat'] = 5

df_train.loc[(60 <= df_train["Age"]) & (df_train["Age"] < 70), 'Age_cat'] = 6

df_train.loc[(70 <= df_train["Age"]), 'Age_cat'] = 7
df_test.loc[df_test["Age"] < 10, 'Age_cat'] = 0 

df_test.loc[(10 <= df_test["Age"]) & (df_test["Age"] < 20), 'Age_cat'] = 1

df_test.loc[(20 <= df_test["Age"]) & (df_test["Age"] < 30), 'Age_cat'] = 2

df_test.loc[(30 <= df_test["Age"]) & (df_test["Age"] < 40), 'Age_cat'] = 3

df_test.loc[(40 <= df_test["Age"]) & (df_test["Age"] < 50), 'Age_cat'] = 4

df_test.loc[(50 <= df_test["Age"]) & (df_test["Age"] < 60), 'Age_cat'] = 5

df_test.loc[(60 <= df_test["Age"]) & (df_test["Age"] < 70), 'Age_cat'] = 6

df_test.loc[(70 <= df_test["Age"]), 'Age_cat'] = 7
def category_age(i):

    if i < 10:

        return 0

    elif i < 20:

        return 1

    elif i < 30:

        return 2

    elif i < 40:

        return 3

    elif i < 50:

        return 4

    elif i < 60:

        return 5

    elif i < 70:

        return 6

    else:

        return 7
df_train['Age_cat_2'] = df_train['Age'].apply(category_age)

df_train.head()
df_train['Age_cat'] == df_train['Age_cat_2']
(df_train['Age_cat'] == df_train['Age_cat_2']).head()
## 모든 값을 비교 모두 T이면 return True, else return False

(df_train['Age_cat'] == df_train['Age_cat_2']).all()
# axis=1 column, axis=0 row

#df_train.drop(["Age", "Age_cat_2"], axis=1, inplace=True)

#df_test.drop(["Age", "Age_cat_2"], axis=1, inplace=True)
df_train.head()
df_test.head()
df_train['Initial'].unique()
df_train['Initial'] = df_train["Initial"].map({'Master': 0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})

df_train.head()
df_test['Initial'] = df_test["Initial"].map({'Master': 0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})

df_test.head()
df_train["Embarked"].unique()
df_train["Embarked"].value_counts()
df_train["Embarked"] = df_train["Embarked"].map({"C":0, "Q":1, "S":2})

df_train.head()
df_test["Embarked"] = df_test["Embarked"].map({"C":0, "Q":1, "S":2})

df_test.head()
df_train.Embarked.isnull().sum()
df_train.Sex.unique()
df_test.Sex.unique()
df_train["Sex"] =df_train["Sex"].map({"female":0, "male":1})

df_test["Sex"] = df_test["Sex"].map({"female":0, "male":1})



df_train.head()
hmap_data = df_train[["Survived", "Pclass", "Sex", "Fare", "Embarked", "FamilySize", "Initial", "Age_cat"]]

hmap_corr = hmap_data.corr(method='spearman')
colormap=plt.cm.BuGn



plt.figure(figsize=(15,10))



sns.clustermap(hmap_corr,linewidths=.1, vmax=1, square=True, linecolor='white', annot=True, annot_kws={"size":16}, cmap=colormap)
colormap=plt.cm.BuGn



plt.figure(figsize=(15,10))



sns.heatmap(hmap_corr, linewidths=.1, vmax=1, square=True, linecolor='white', annot=True, annot_kws={"size":16}, cmap=colormap)
pd.get_dummies(data=df_train,columns=["Initial"], prefix='Initial').head()
df_train = pd.get_dummies(data=df_train,columns=["Initial"], prefix='Initial')

df_test = pd.get_dummies(data=df_test,columns=["Initial"], prefix='Initial')
df_train = pd.get_dummies(data=df_train,columns=["Embarked"], prefix='Embarked')

df_test = pd.get_dummies(data=df_test,columns=["Embarked"], prefix='Embarked')
df_train.head()
df_train.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"], axis=1, inplace=True)

df_test.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"], axis=1, inplace=True)
df_train.head()
df_test.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split
X_train = df_train.drop(["Survived", "Fare.log"], axis=1,).values

target_label = df_train["Survived"].values

X_test = df_test.values
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)
model = RandomForestClassifier()

model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)
prediction[:10]
print('accuracy {:.2f}%'.format(100 * metrics.accuracy_score(prediction, y_vld)))
(prediction == y_vld).sum() / prediction.shape[0]
from pandas import Series

feature_importance = model.feature_importances_

feat_imp = Series(feature_importance, index=df_train.drop(["Survived", "Fare.log"], axis=1,).columns)
df_train.head()
df_test.head()
plt.figure(figsize=(8,8))

feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel("Feature importance")

plt.ylabel("Feature")
submission = pd.read_csv("../input/gender_submission.csv")
submission.head()
prediction = model.predict(X_test)
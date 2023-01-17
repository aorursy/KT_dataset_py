# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

import sklearn.preprocessing as preprocessing

from sklearn import linear_model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("../input/train.csv")

print(data_train.head(5))
data_train.info()
data_train.describe()
fig = plt.figure()

fig.set(alpha=0.2)



plt.subplot2grid((3,5), (0,0))

data_train.Survived.value_counts().plot(kind='bar')

plt.title(u"Survived status (1 means survivied)")

plt.ylabel(u"persons")



plt.subplot2grid((3,5), (0,2))

data_train.Pclass.value_counts().plot(kind='bar')

plt.ylabel(u"persons")

plt.title(u"distribution of pclass")



plt.subplot2grid((3, 5), (0, 4))

plt.scatter(data_train.Survived, data_train.Age)

plt.ylabel(u"age")

plt.grid(b=True, which="major", axis="y")

plt.title(u"Survived status by age(1 means survivied)")



plt.subplot2grid((3, 5), (2, 0), colspan=2, rowspan=2)

data_train.Age[data_train.Pclass == 1].plot(kind='kde')   

data_train.Age[data_train.Pclass == 2].plot(kind='kde')

data_train.Age[data_train.Pclass == 3].plot(kind='kde')

plt.xlabel(u"age")# plots an axis lable

plt.ylabel(u"density") 

plt.title(u"age distribution by pclass")

plt.legend((u'first class', u'second class',u'third class'),loc='best') # sets our legend for our graph.





plt.subplot2grid((3,5),(2,3))

data_train.Embarked.value_counts().plot(kind='bar')

plt.title(u"persons of differnt port")

plt.ylabel(u"persons")  

plt.show()
fig2 = plt.figure()

fig2.set(alpha=0.2)



Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()



df = pd.DataFrame({u'Survived':Survived_1, u'non-Survived': Survived_0})

df.plot(kind='bar', stacked=True)

plt.title('Survived status of different Pclass')

plt.xlabel("Pclass")

plt.ylabel("Persons")

plt.show()
fig = plt.figure()

fig.set(alpha=0.2)



Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()

Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

df = pd.DataFrame({'Male': Survived_m, 'Female': Survived_f})

df.plot(kind='bar', stacked=True)

plt.title('Survived status by different sex')

plt.xlabel('Sex')

plt.ylabel('Persons')

plt.show()
Survived_m
fig = plt.figure()

fig.set(alpha=0.65)

plt.title("Survived status by pclass and sex")



ax1 = fig.add_subplot(141)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')

ax1.set_xticklabels(['Survived', 'non-Survived'], rotation = 0)

ax1.legend(['female/high class'], loc='best')



ax2=fig.add_subplot(142, sharey=ax1)

data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')

ax2.set_xticklabels(['Survived', 'non-Survived'], rotation=0)

plt.legend(['female/low class'], loc='best')



ax3=fig.add_subplot(143, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')

ax3.set_xticklabels(['Survived', 'non-Survived'], rotation=0)

plt.legend(["male/high class"], loc='best')



ax4=fig.add_subplot(144, sharey=ax1)

data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')

ax4.set_xticklabels(['Survived', 'non-Survived'], rotation=0)

plt.legend(["male/low class"], loc='best')



plt.show()
fig = plt.figure()

fig.set(alpha = 0.2)



Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()

Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()

df = pd.DataFrame({'Survived': Survived_0, 'non-Survived': Survived_1})

df.plot(kind='bar', stacked=True)

plt.title('Survived status by ports')

plt.xlabel('Survived status')

plt.ylabel('Person')

plt.show()
g = data_train.groupby(['SibSp', 'Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

print(df)
g = data_train.groupby(['Parch', 'Survived'])

df = pd.DataFrame(g.count()['PassengerId'])

print(df)
def set_missing_age(df):

    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    

    known_age = age_df[age_df.Age.notnull()].as_matrix()

    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    

    y = known_age[:, 0]

    x = known_age[:, 1:]

    

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)

    rfr.fit(x, y)

    

    predictedAges = rfr.predict(unknown_age[:, 1::])

    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    

    return df, rfr
def set_Cabin_type(df):

    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"

    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"

    return df
data_train, rfr = set_missing_age(data_train)

data_train = set_Cabin_type(data_train)
data_train.info()
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')



df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)



df
scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))

df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

df
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

train_np = train_df.values



y = train_np[:, 0]

X = train_np[:, 1:]



clf = linear_model.LogisticRegression(solver='liblinear', C=1.0, penalty='l1', tol=1e-6)

clf.fit(X, y)



clf
data_test = pd.read_csv("../input/test.csv")

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

# 接着我们对test_data做和train_data中一致的特征变换

# 首先用同样的RandomForestRegressor模型填上丢失的年龄

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

null_age = tmp_df[data_test.Age.isnull()].values

# 根据特征属性X预测年龄并补上

X = null_age[:, 1:]

predictedAges = rfr.predict(X)

data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges



data_test = set_Cabin_type(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')





df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)

df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

df_test.head()
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})

result.to_csv("titanic_logistic_regression_predictions.csv", index=False)
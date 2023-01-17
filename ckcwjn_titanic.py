import re

import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import learning_curve



import seaborn as sns

sns.set_style('whitegrid')



import warnings

warnings.filterwarnings('ignore')



###pandas显示列不限制###

pd.options.display.max_columns = None



###sklearn###

from sklearn.cluster import KMeans

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



df = pd.concat([train, test])

df.reset_index(inplace=True)

df.drop('index', axis=1, inplace=True)

sns.set_style('whitegrid')

print(df.head())

print('\n\n##########################################################################\n\n')

df.info()
###Embarked、Fare缺失值少，使用众数补充###

df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values

#df.Fare[df.Fare.isnull()] = df.Fare.dropna().mode().values



###Cabin缺失的使用‘UO’进行填充###

df['Cabin'] = df.Cabin.fillna('U0') # 两种写法 df.Cabin[df.Cabin.isnull()]='U0'



###Age###

##使用均值加减标准差填充

# average_age = df["Age"].mean()

# std_age = df["Age"].std()

# count_nan_age = df["Age"].isnull().sum()

# rand = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# df['Age'][df.Age.isnull()] = rand



df.info()



##重新赋值train和test数据

train = df[:train.shape[0]]

print(train.shape[0])

test = df[train.shape[0]:]
###特征间相关性###

corrdf = train.corr()

plt.subplots(figsize=(9, 9)) # 设置画面大小

sns.heatmap(corrdf, annot=True, vmax=1, square=True, cmap="Blues")

plt.show()
###Pclass###

print(train.groupby(['Pclass','Survived'])['Pclass'].count()) ##Pclass与Survived关系

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

plt.show()
###Age###

##年龄分布情况

plt.figure(figsize=(12,5))

plt.subplot(121)

train['Age'].hist(bins=70)

plt.xlabel('Age')

plt.ylabel('Num')

plt.subplot(122)

train.boxplot(column='Age', showfliers=False)



##不同年龄下生存与非生存的分布情况

facet = sns.FacetGrid(df, hue="Survived",aspect=4)

facet.map(sns.kdeplot,'Age',shade= True)

facet.set(xlim=(0, df['Age'].max()))

facet.add_legend()

    

##不同年龄存活率

# fig, axis1 = plt.subplots(1,1,figsize=(18,4))

# train['Age_int'] = train['Age'].astype(int)

# average_age = train[['Age_int', 'Survived']].groupby(['Age_int'],as_index=False).mean()

# sns.barplot(x='Age_int', y='Survived', data=average_age)



##不同年龄存活人数

# fig, axis1 = plt.subplots(1,1,figsize=(18,4))

# age = train[['Survived','Age_int']][train['Survived']==1].groupby(['Age_int'],as_index=False).count()

# sns.barplot(x='Age_int', y='Survived', data=age)



plt.show() 



##年龄详细数据描述

train['Age'].describe(include=['O'])
###SibSp###

##分成有无兄弟

sibsp_df = train[train['SibSp'] != 0]

no_sibsp_df = train[train['SibSp'] == 0]



plt.figure(figsize=(10,5))

plt.subplot(121)

sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')

plt.xlabel('sibsp')



plt.subplot(122)

no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')

plt.xlabel('no_sibsp')



plt.show()
###Parch###

##分成有无父母子女

parch_df = train[train['Parch'] != 0]

no_parch_df = train[train['Parch'] == 0]



plt.figure(figsize=(10,5))

plt.subplot(121)

parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')

plt.xlabel('parch')



plt.subplot(122)

no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')

plt.xlabel('no_parch')



plt.show()
###Embarked###

##泰坦尼克号从英国的南安普顿港出发，途径法国瑟堡和爱尔兰昆士敦，那么在昆士敦之前上船的人，有可能在瑟堡或昆士敦下船，这些人将不会遇到海难。

sns.countplot('Embarked', hue='Survived', data=train)

plt.title('Embarked and Survived')
###Fare###

##Fare项在测试数据中缺少一个值，按照一二三等舱各自的均价来填充

df['Fare'] = df[['Fare']].fillna(df.groupby('Pclass').transform(np.mean))



##通过对Ticket数据的分析，部分票号数据有重复，同时结合亲属人数及名字的数据，和票价船舱等级对比

##购买的票中有家庭票和团体票，将团体票的票价分配到每个人的头上

df['Group_Ticket'] = df['Fare'].groupby(by=df['Ticket']).transform('count')

df['Fare'] = df['Fare'] / df['Group_Ticket']

df.drop(['Group_Ticket'], axis=1, inplace=True)
###Pclass###

pclass_dummies  = pd.get_dummies(df['Pclass'])

pclass_dummies.columns = ['Class_1','Class_2','Class_3']

df.drop(['Pclass'],axis=1,inplace=True)

df = df.join(pclass_dummies)

df.info()
###Name###

##提取称呼

df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

##映射称呼

title_Dict = {}

title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))

title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))

title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))

title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))

title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))

title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

df['Title'] = df['Title'].map(title_Dict)

df['Title'] = pd.factorize(df['Title'])[0]

##映射后每种称呼的平均年龄

# Title_squage = df[['Title', 'Age']].groupby(['Title'],as_index=False).mean()

# print(Title_squage)



df.head()
###Sex###

df['Sex'][df['Sex'] == 'male'] = 1

df['Sex'][df['Sex'] == 'female'] = 0

df['Sex'] = df['Sex'].astype(int)
###Parch and SibSp###

df['family_size'] = df['Parch'] + df['SibSp']

print(df.groupby(['family_size','Survived'])['family_size'].count())

df[['family_size','Survived']].groupby(['family_size']).mean().plot.bar()

plt.show()





def family_size_category(family_size):

    if family_size <= 0:

        return 'Single'

    elif family_size <= 3:

        return 'Small_Family'

    else:

        return 'Large_Family'



df['Family_Size'] = df['family_size'].map(family_size_category)



df['Family_num']  =pd.factorize(df['Family_Size'])[0]

#Family_Size_dummies.columns = ['single','small','larger']

#df = df.join(Family_Size_dummies)

df.drop(['family_size','Family_Size'],axis=1,inplace=True)

df.info()
###Age###

##机器学习预测

missing_age_df = pd.DataFrame(df[

 ['Age','Title', 'Family_num','Fare']])

missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]

missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]



X = missing_age_train.values[:,1:]

Y = missing_age_train.values[:,0]

RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)

RFR.fit(X,Y)

predictAges = RFR.predict(missing_age_test.values[:,1:])

df.loc[df['Age'].isnull(), ['Age']]= predictAges



##Scaling到-1~1

scaler = preprocessing.StandardScaler()

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))

df['Age_scaled'].head()
###Embarked###

##dummies 

embark_dummies  = pd.get_dummies(df['Embarked'])

df = df.join(embark_dummies)

df.drop(['Embarked'], axis=1,inplace=True)

embark_dummies = df[['S', 'C', 'Q']]

embark_dummies.head()
###Fare###
df.info()

df_backend = df.drop(['Age', 'Cabin','Name','Parch','PassengerId','SibSp','Survived','Ticket'],axis=1,inplace=False)

df_backend.info()
X = df_backend[:train.shape[0]]

Y = df[:train.shape[0]]['Survived']

random_forest = RandomForestClassifier(oob_score=True, n_estimators=1000)

random_forest.fit(X, Y)



X_test = df_backend[train.shape[0]:]

Y_pred = random_forest.predict(X_test)

print(random_forest.score(X, Y))



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



title = "Learning Curves"

plot_learning_curve(RandomForestClassifier(oob_score=True, n_estimators=3000), title, X, Y, cv=None,  n_jobs=4)

plt.show()



submission = pd.DataFrame({

    "PassengerId": df[train.shape[0]:]["PassengerId"],

    "Survived": Y_pred.astype(int)

})

submission.to_csv('submission.csv', index=False)
import numpy as np  # 科学计算工具包

import pandas as pd  # 数据分析工具包

import matplotlib.pyplot as plt # 图表绘制工具包

import seaborn as sns # 基于 matplot, 导入 seaborn 会修改默认的 matplotlib 配色方案和绘图样式，这会提高图表的可读性和美观性



# 算法库

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB



# 在 jupyter notebook 里面显示图表

%matplotlib inline 
test_df = pd.read_csv("../input/titanic/test.csv")

train_df = pd.read_csv("../input/titanic/train.csv")
train_df.head()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train_total = train_df.isnull().sum().sort_values(ascending=False)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = round(percent_1,1).sort_values(ascending=False)

train_miss_data = pd.concat([train_total,percent_2],axis=1,keys=['total','%'])

train_miss_data.head()
test_total = test_df.isnull().sum().sort_values(ascending=False)

percent_1 = test_df.isnull().sum()/test_df.isnull().count()*100

percent_2 = round(percent_1,1).sort_values(ascending=False)

test_miss_data = pd.concat([test_total,percent_2],axis=1,keys=['total','%'])

test_miss_data.head()
train_df.columns.values
# 按性别筛选出数据

women = train_df[train_df['Sex']=='female']

men = train_df[train_df['Sex']=='male']



# 在性别的基础上筛选出存活和未存活的数据



# 选出存活的数据

F_survived = women[women['Survived']==1]

M_survived = men[men['Survived']==1]



# 选出未存活的数据

F_not_surv = women[women['Survived']==0]

M_not_surv = men[men['Survived']==0]



F_survived.head()
# 每种数据去除 Age 缺失值

print('去除前，Female survived null', F_survived.Age.isnull().sum())



# 去除 Age 缺失值

F_survived.Age.dropna()

M_survived.Age.dropna()

F_not_surv.Age.dropna()

M_not_surv.Age.dropna()



print('取出后，Female survived null',F_survived.Age.dropna().isnull().sum())
sns.set() # 声明使用 Seaborn 样式



fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(16,8)) # 创建一个 Figure, 子图为1行，2列

survived = 'survived' # 图例 label 

not_survived = 'not survived' # 图例 label



ax = sns.distplot(F_survived.Age.dropna(),bins=18,ax=axes[0],kde=False)

ax = sns.distplot(F_not_surv.Age.dropna(),bins=40,ax=axes[0],kde=False)

ax.legend([survived,not_survived]) # 图例 label 放置位置1

ax.set_title('Female')



ax = sns.distplot(M_survived.Age.dropna(),bins=18,ax=axes[1],label=survived,kde=False) # 图例 label 放置位置2

ax = sns.distplot(M_not_surv.Age.dropna(),bins=40,ax=axes[1],label=not_survived,kde=False)

ax.legend()

ax.set_title('Male')
import matplotlib.pyplot as plt #导入 matplotlib.pyplot，并简写成plt

import seaborn as sns

import numpy as np  #导入numpy包，用于生成数组

import pandas as pd #导入pandas包，用于数据分析

#IPython notebook中的魔法方法，这样每次运行后可以直接得到图像，不再需要使用plt.show()

%matplotlib inline  



g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
import matplotlib.pyplot as plt #导入 matplotlib.pyplot，并简写成plt

import seaborn as sns

import numpy as np  #导入numpy包，用于生成数组

import pandas as pd #导入pandas包，用于数据分析

#IPython notebook中的魔法方法，这样每次运行后可以直接得到图像，不再需要使用plt.show()

%matplotlib inline  



sns.barplot(x='Sex', y='Survived', data=train_df)
grid = sns.FacetGrid(train_df, row='Embarked',height=2.2, aspect=1.6) 

grid.map(sns.pointplot, 'Pclass', 'Survived','Sex',palette='deep',hue_order=['female','male'],order=[1,2,3]) 

grid.add_legend()
sns.barplot(x='Pclass', y='Survived', data=train_df)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
# 或者用这个

grid = sns.FacetGrid(train_df,hue='Survived',row='Pclass')

grid.map(plt.hist,"Age",bins=20)

grid.add_legend()
data = [train_df, test_df]  # 训练集和测试集

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0

    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1

    dataset['not_alone'] = dataset['not_alone'].astype(int)

train_df['not_alone'].value_counts()

df1 = pd.DataFrame(np.random.rand(8,4),index=list('abcdefgh'),columns=['A','B','C','D'])

print(df1)



df1.loc[df1['A']<0.5,'小于0.5'] = 1

df1.loc[df1['A']>0.5,'小于0.5'] = 0

print(df1)
df1 = pd.DataFrame(np.random.rand(8,4),index=list('abcdefgh'),columns=['A','B','C','D'])

print(df1)



df1.loc[df1['A']<0.5,'小于0.5'] = 1

print(df1)

print(df1['小于0.5'].value_counts())

df1.loc[df1['A']>0.5,'小于0.5'] = 0

print(df1['小于0.5'].value_counts(normalize=True))
df1['小于0.5'].astype(int).value_counts()
grid = sns.catplot('relatives','Survived', data=train_df, kind='point',aspect = 2.5)
# 合并训练集和测试集

titanic = train_df.append(test_df, ignore_index=True)



# 保存测试集的 PassengerId 用于最后提交

passengerId = test_df.PassengerId



# 创建索引，后期用于分开数据集

train_idx = len(train_df)

test_idx = len(titanic) - len(test_df)



print(titanic.info())
#train_df = train_df.drop(['PassengerId'], axis=1)
# 正则测试

import re



test = 'Braund,the Countess. Owen Harris'

pattern =re.compile(",(.+)\.")

print(pattern.search(test).group(1))
# 训练集

train_df['Title'] = train_df['Name'].map(lambda x:(re.compile(",(.+?)\.").search(x).group(1)).strip())

print(list(train_df['Title'].drop_duplicates()))



# 测试集

test_df['Title'] = test_df['Name'].map(lambda x:(re.compile(",(.+?)\.").search(x).group(1)).strip())

print(list(test_df['Title'].drop_duplicates()))
Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty",

    "Dona":"Royalty"

}



titanic['Title'] = titanic['Name'].map(lambda x:(re.compile(",(.+?)\.").search(x).group(1)).strip())

titanic['Title'] = titanic['Title'].map(Title_Dictionary)



# 查看缺失值

print(titanic['Title'].isnull().sum())

#titanic[titanic['Title'].isnull() == True]
# 相同尊称的人数

titanic['Title'].value_counts()
grouped = titanic.groupby(['Sex','Pclass', 'Title'])  

grouped["Age"].median()
titanic["Age"] = grouped["Age"].apply(lambda x: x.fillna(x.median()))



# 查看处理后的情况

titanic.info()
# 训练集

train_df['Cabin'] = train_df['Cabin'].fillna("U0") # 将缺失值填充为 “U0”  表示 Unknow

# 正则获取夹板号 并 使用 drop_duplicates() 去重

print(list(train_df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group()).drop_duplicates()))
# 测试集

test_df['Cabin'] = test_df['Cabin'].fillna("U0") # 将缺失值填充为 “U0”  表示 Unknow

# 正则获取夹板号 并 使用 drop_duplicates() 去重

print(list(test_df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group()).drop_duplicates()))
import re

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}



titanic['Cabin'] = titanic['Cabin'].fillna("U0") # 没有船舱号 将缺失值填充为 “U0”

titanic['Deck'] = titanic['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())#正则获取夹板号

titanic['Deck'] = titanic['Deck'].map(deck) # 通过字典用 map 映射夹板号为数字

titanic['Deck'] = titanic['Deck'].fillna(0) # 没有夹板号 将缺失值填充为 “0”

titanic['Deck'] = titanic['Deck'].astype(int) # 将dateframe某一列的数据类型转化为整数型



# 处理完删除 cabin 特征

# train_df = train_df.drop(['Cabin'], axis=1)

# test_df = test_df.drop(['Cabin'], axis=1)



# 处理后的情况

titanic.info()
# 用 value_counts() 获取众数

print(titanic['Embarked'].value_counts()) # 默认降序



# 获取行标签

print(titanic['Embarked'].value_counts().index)



# 获取第一行的行标签

print(titanic['Embarked'].value_counts().index[0])
# 用 mode() 获取众数

print(titanic['Embarked'].mode())

print(titanic['Embarked'].mode().iloc[0])
# 用众数填充 Embarked

titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode().iloc[0])



# 用中位数填充 Fare

titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())



# 用‘U’填充 Cabin

titanic['Cabin'] = titanic['Cabin'].fillna('U')



# 看处理后结果

titanic.info()
# 同行家庭数 (包括乘客本身)

titanic['FamilySize'] = titanic['Parch'] + titanic['SibSp'] + 1
titanic['Deck'] = titanic['Cabin'].map(lambda x: x[0])

titanic['Deck']
titanic.head()
# 将性别转化为整数形式

titanic['Sex'] = titanic['Sex'].map({"male": 0, "female":1})



# 类别变量转化为dummy 变量

pclass_dummies = pd.get_dummies(titanic.Pclass, prefix="Pclass")

title_dummies = pd.get_dummies(titanic.Title, prefix="Title")

deck_dummies = pd.get_dummies(titanic.Deck, prefix="Deck")

embarked_dummies = pd.get_dummies(titanic.Embarked, prefix="Embarked")



# 合并 dummy 列和原数据集

titanic_dummies = pd.concat([titanic, pclass_dummies, title_dummies, deck_dummies, embarked_dummies], axis=1)



# 删除类别字段

titanic_dummies.drop(['Pclass', 'Title', 'Cabin','Deck','Embarked', 'Name', 'Ticket'], axis=1, inplace=True)



titanic_dummies.head()
# 分离训练集和测试集

train = titanic_dummies.iloc[ :train_idx]

test = titanic_dummies.iloc[test_idx: ]



# 转化 Survived 特征为整数型

train.Survived = train.Survived.astype(int)



# 训练集分成 X 和 Y(目标变量：Survived) 

x_train = train.drop('Survived', axis=1).values 

y_train = train.Survived.values



# 测试集删除，训练集的目标变量：Survived

x_test = test.drop('Survived', axis=1).values



print(train.head())

print(train.info())
# classifier models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



# create param grid object 

forrest_params = dict(     

    max_depth = [n for n in range(9, 14)],     

    min_samples_split = [n for n in range(4, 11)], 

    min_samples_leaf = [n for n in range(2, 5)],     

    n_estimators = [n for n in range(10, 60, 10)],

)



# instantiate Random Forest model

forrest = RandomForestClassifier()



# build and fit model 

forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5) 

forest_cv.fit(x_train, y_train)



print("Best score: {}".format(forest_cv.best_score_))

print("Optimal params: {}".format(forest_cv.best_estimator_))
# random forrest prediction on test set

forrest_pred = forest_cv.predict(x_test)



# dataframe with predictions

kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': forrest_pred})

# save to csv

kaggle.to_csv('submission.csv', index=False)

print("Submitted successfully")
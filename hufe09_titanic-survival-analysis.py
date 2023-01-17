import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns





pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全

pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行

%matplotlib inline

sns.set_style('darkgrid')
# 数据加载

df = pd.read_csv('../input/train.csv')

df.head()
# 数据形状

df.shape
# 以下是一些摘要统计数据

df.describe()
df.info()
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

df.head()
df.hist(figsize=(10, 8))
df[df.Age.isnull()].hist(figsize=(10, 8))
df.fillna(df.mean(), inplace=True)

df.info()
df[df.Embarked.isnull()]
df['Embarked'].value_counts()
df['Embarked'].fillna('S', inplace=True)
df.info()
survived = df.Survived == True

died = df.Survived == False
df.Fare[survived].mean()
df.Fare[died].mean()
df.Fare[survived].hist(alpha=0.5, bins=20, label="survived")

df.Fare[died].hist(alpha=0.5, bins=20, label="died")

plt.legend()
df.Age[survived].hist(alpha=0.5, bins=20, label="survived")

df.Age[died].hist(alpha=0.5, bins=20, label="died")

plt.legend()
df.groupby('Pclass').Survived.mean()
df.groupby('Pclass').Survived.mean().plot(kind="bar")
df.Sex.value_counts()
df.groupby('Sex').Survived.mean()
df.groupby('Sex').Survived.mean().plot(kind="bar")
df.groupby('Sex')['Pclass'].value_counts()
df.query('Sex == "female"')['Fare'].median(), df.query(

    'Sex == "male"')["Fare"].median()
df.groupby(['Pclass', 'Sex']).Survived.mean().plot(kind='bar')
df.SibSp[survived].value_counts().plot(

    kind='bar', alpha=0.5, color='green', label='survived')

df.SibSp[died].value_counts().plot(

    kind='bar', alpha=0.5, color='red', label='died')

plt.legend()
df.Parch[survived].value_counts().plot(

    kind='bar', alpha=0.5, color='green', label='survived')

df.Parch[died].value_counts().plot(

    kind='bar', alpha=0.5, color='red', label='died')

plt.legend()
df.Embarked[survived].value_counts()
df.Embarked[died].value_counts()
df.Embarked[survived].value_counts().plot(

    kind='bar', alpha=0.5, color='green', label='survived')

df.Embarked[died].value_counts().plot(

    kind='bar', alpha=0.5, color='red', label='died')

plt.legend()
# encoding=utf-8

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_iris

# 准备数据集

iris = load_iris()

# 获取特征集和分类标识

features = iris.data

labels = iris.target

# 随机抽取 33% 的数据作为测试集，其余为训练集

train_features, test_features, train_labels, test_labels = train_test_split(

    features, labels, test_size=0.33, random_state=0)

# 创建 CART 分类树

clf = DecisionTreeClassifier(criterion='gini')

# 拟合构造 CART 分类树

clf = clf.fit(train_features, train_labels)

# 用 CART 分类树做预测

test_predict = clf.predict(test_features)

# 预测结果与测试集结果作比对

score = accuracy_score(test_labels, test_predict)

print("CART 分类树准确率 %.4lf" % score)
# 最后利用graphviz库打印出决策树图

from sklearn import tree

import graphviz

dot_data = tree.export_graphviz(

    clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)

graph
import pandas as pd

pd.set_option('display.max_columns', 100)  # 设置显示数据的最大列数，防止出现省略号…，导致数据显示不全

pd.set_option('expand_frame_repr', False)  # 当列太多时不自动换行
# 数据加载

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

# 数据探索

train_data.info()
train_data.head()
test_data.tail()
train_data.describe()
train_data.describe(include=['O'])
# 使用平均年龄来填充年龄中的 nan 值

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

# 使用票价的均值填充票价中的 nan 值

train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
sum(train_data['Embarked'].value_counts())

train_data['Embarked'].value_counts()
# 使用登录最多的港口来填充登录港口的 nan 值

train_data['Embarked'].fillna('S', inplace=True)

test_data['Embarked'].fillna('S', inplace=True)
x = test_data.isnull().sum()

y = train_data.isnull().sum()

print(x, y)
# 特征选择

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

train_features = train_data[features]

train_labels = train_data['Survived']

test_features = test_data[features]
from sklearn.feature_extraction import DictVectorizer

dvec = DictVectorizer(sparse=False)

train_features = dvec.fit_transform(train_features.to_dict(orient='record'))
dvec.feature_names_
train_data.info()
test_data.info()
from sklearn.tree import DecisionTreeClassifier

# 构造 ID3 决策树

clf = DecisionTreeClassifier(criterion='entropy')

# 决策树训练

clf.fit(train_features, train_labels)
test_features = dvec.transform(test_features.to_dict(orient='record'))

# 决策树预测

pred_labels = clf.predict(test_features)
# 得到决策树准确率

acc_decision_tree = round(clf.score(train_features, train_labels), 6)

print(u'score 准确率为 %.4lf' % acc_decision_tree)
import numpy as np

from sklearn.model_selection import cross_val_score

# 使用 K 折交叉验证 统计决策树准确率

print(u'cross_val_score 准确率为 %.4lf' %

      np.mean(cross_val_score(clf, train_features, train_labels, cv=10)))
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": pred_labels

    })

submission.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv').shape
# 最后利用graphviz库打印出决策树图

from sklearn import tree

import graphviz

dot_data = tree.export_graphviz(

    clf, out_file=None, filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)

graph.render("titanic_id3")

graph
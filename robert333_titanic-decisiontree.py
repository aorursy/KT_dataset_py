import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.feature_extraction import DictVectorizer
train_data = pd.read_csv('../input/titanic/train.csv');

test_data = pd.read_csv('../input/titanic/test.csv');
print('查看数据信息：列名、非空个数、类型等');

print(train_data.info());

print('-'*30);

print('查看数据摘要');

print(train_data.describe());

print('-'*30);

print('查看离散数据分布');

print(train_data.describe(include=['O']));

print('-'*30);

print('查看前5条数据');

print(train_data.head());

print('-'*30);

print('查看后5条数据');

print(train_data.tail());
#使用平均年龄来填充年龄中的nan值

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True);

print(train_data['Age']);

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True);

print(test_data['Age']);
# 使用票价的均值填充票价中的nan值

train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True);

print(train_data['Fare']);

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True);

print(test_data['Fare']);
print(train_data['Embarked'].value_counts());

# 使用登陆最多的港口来填充登陆港口的nan值

train_data['Embarked'].fillna('S', inplace=True);

test_data['Embarked'].fillna('S', inplace=True);

print(train_data['Embarked']);
# 特征选择

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];

train_features = train_data[features];

train_labels = train_data['Survived'];

test_features = test_data[features];

print('特征值');

print(train_features);
print(train_features.iloc[0:2, :]);

a = train_features.iloc[0:2, :];

b = a.to_dict(orient='record');

print(b);

print('-'*40);

dvec = DictVectorizer(sparse=False);

c = dvec.fit_transform(b);

print(c);

print(dvec.feature_names_)
dvec = DictVectorizer(sparse=False);

train_features = dvec.fit_transform(train_features.to_dict(orient='record'));

print(train_features);

print(dvec.feature_names_);
# 构造ID3决策树

clf = DecisionTreeClassifier(criterion='entropy');

# 决策树训练

clf.fit(train_features, train_labels);
test_features = dvec.transform(test_features.to_dict(orient='record'));

# 决策树预测

pred_labels = clf.predict(test_features);
# 得到决策树准确率（基于训练集）

acc_decision_tree = round(clf.score(train_features, train_labels), 6);

print(u'score准确率为%.4lf' % acc_decision_tree);



# 使用k折交叉验证 统计决策树准确率

print(u'cross_val_score准确率为%.4lf' % np.mean(cross_val_score(clf, train_features, train_labels, cv=10)));
# 得到决策树准确率（基于测试集）

acc_decision_tree = round(clf.score(train_features, train_labels), 6);

print(acc_decision_tree);
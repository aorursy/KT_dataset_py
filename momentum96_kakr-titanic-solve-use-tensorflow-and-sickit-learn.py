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
# pd.read_csv('../input/sample_submission.csv')
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(10)
test.head()
train.shape
test.shape
train.info()
test.info()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(train.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.OrRd, linecolor='white', annot=True)
plt.show()
train.isnull().sum()
test.isnull().sum()
# 산 사람과 죽은 사람의 각 피쳐 특징 확인
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts() # Survived 값이 1인 행들의 feature열 수 count
    dead = train[train['Survived'] == 0][feature].value_counts() # Survived 값이 0인 행들의 feature열 수 count
    df = pd.DataFrame([survived, dead]) # 산사람과 죽은사람으로 나누어서 DataFrame으로 저장
    df.index = ['Survived', 'Dead'] # index명 지정
    df.plot(kind='bar', stacked=True, figsize=(10, 5)) # bar chart 그리기
    plt.show()
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
train.head(10)
train_test_data = [train, test] # train data와 test data 결합

# train_test_data의 Name필드에서 Title을 뽑음
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
# train data의 타이틀 종류 및 인원 확인
train['Title'].value_counts()
# test data의 타이틀 종류 및 인원 확인
test['Title'].value_counts()
# 타이틀별로 Mr는 0, Miss는 1, Mrs는 2, 그 외 나머지는 3으로 매핑

title_mapping = {"Mr":0, "Miss":1, "Mrs":2,
                 "Master":3, "Dr":3, "Rev":3, "Major":3, "Col":3, "Mlle":3, "Lady":3,
                 "Mme":3, "Sir":3, "Jonkheer":3, "Ms":3, "Capt":3, "Don":3, "Dona":3,
                "Countess":3}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
train.head(10)
bar_chart('Title')
# 데이터셋 중 필요없는 피쳐 삭제
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
train.head()
sex_mapping = {"male":0, "female":1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
bar_chart('Sex')
# Age 필드의 NaN값을 Title 그룹별의 나이 중간값으로 채움
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)
# train data의 나이에 따른 생사 확인
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()
# 나이대별 생사 확인
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(20, 30)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(30, 40)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(40, 60)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.xlim(60)
# 나이대에 따라 그룹 나눔
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4
train.head()
bar_chart('Age')
Pclass1 = train[train['Pclass'] == 1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass'] == 2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass'] == 3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10, 5))
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train['Embarked'].isnull().sum()
embarked_mapping = {"S":0, "C":1, "Q":2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
# 등급별 중간값을 NaN값에 넣어줌
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()

plt.show()
for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 2,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 4,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 6
train.head()
train.Cabin.value_counts()
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind = 'bar', stacked=True, figsize=(10, 5))
train.Cabin.value_counts()
# scaling
# 머신 러닝 모델은 값의 차이가 클 수록 더 큰 의미를 부여하기 때문에 값을 스케일링 해줌
cabin_mapping = {"A":0, "B":0.7, "C":1.4, "D":2.1, "E":2.8, "F":3.5, "G":4.2, "T":4.9}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
train.head(10)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'FamilySize', shade=True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)
family_mapping = {1:0, 2:0.5, 3:1.0, 4:1.5, 5:2.0, 6:2.5, 7:3.0, 8:3.5, 9:4.0, 10:4.5, 11:5.0}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
corr = train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
train.head()
# 필요없는 항목 drop
features_drop = ['Ticket', 'SibSp', 'Parch', 'Cabin']
train.drop(features_drop, axis=1, inplace=True)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)
# 모델 입력데이터 구성을 위한 train_data 셋 구성
#train_data = 입력 , target = 출력
train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape
train_data.head(10)
test = test.drop(['PassengerId'], axis=1)
test.head(10)
plt.figure(figsize=(8, 8))
sns.heatmap(train.corr(), linewidths=0.1, vmax=0.5, cmap=plt.cm.gist_heat, linecolor='white', annot=True)
plt.show()
import tensorflow as tf
tf.set_random_seed(777)

train_x = train_data
df = pd.DataFrame(target) # 산사람과 죽은사람으로 나누어서 DataFrame으로 저장
df.columns = ['Survived'] # index명 지정
train_y = df
test_x = test
test_y = pd.read_csv('../input/gender_submission.csv')
df = pd.DataFrame(test_y['Survived'])
df.columns = ['Survived']
test_y = df
train_x.shape, train_y.shape, test_x.shape, test_y.shape
X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.get_variable("W", shape=[7, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]), name='bias')
H = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y) * tf.log(1-H))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

predicted = tf.cast(H > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
import time
startTime = time.time()
acc_nn = 0.0
nn_predicted_result = []
feature = train_data.columns.tolist()
Weight = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:train_x, Y:train_y})
        if step % 1000 == 0:
            print(step, cost_val)
                  
    print('-----------------------------')
    print('train_data = ', len(train_x), 'test_data = ', len(test_x))
    Weight = sess.run(W)
    
    for i in range(len(feature)):
        print('W', i, '=', Weight[i], ', feature =', feature[i])

    h, c, y, a = sess.run([H, predicted, Y, accuracy], feed_dict={X:test_x, Y:test_y})
    
acc_nn = round(a * 100, 2)
print("accuracy : ", acc_nn)
nn_predicted_result = c
nn_predicted_result = nn_predicted_result.reshape([-1])
nn_predicted_result = nn_predicted_result.astype(int)

endTime = time.time()
print(endTime - startTime, " sec")
Weight = abs(Weight)

from pandas import Series

Weight = np.reshape(Weight, [-1])

tensorflow_feat_imp = Series(Weight, index = feature)
plt.figure(figsize=(10, 10))
tensorflow_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Weight importance')
plt.ylabel('Feature')
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
#Logistic Regression 모델

logreg = LogisticRegression()
logreg.fit(train_x, train_y)
Y_pred = logreg.predict(test_x)
acc_log = round(logreg.score(train_x, train_y) * 100 , 2)
acc_log
# Support Vector Machines 모델

svc = SVC()
svc.fit(train_x, train_y)
Y_pred = svc.predict(test_x)
acc_svc = round(svc.score(train_x, train_y) * 100, 2)
acc_svc
#K Neighbors Classifier 모델

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_x, train_y)
Y_pred = knn.predict(test_x)
acc_knn = round(knn.score(train_x, train_y) * 100, 2)
acc_knn
# Gaussian Naive Bayes 모델

gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
Y_pred = gaussian.predict(test_x)
acc_gaussian = round(gaussian.score(train_x, train_y) * 100, 2)
acc_gaussian
# Perceptron 모델

perceptron = Perceptron()
perceptron.fit(train_x, train_y)
Y_pred = perceptron.predict(test_x)
acc_perceptron = round(perceptron.score(train_x, train_y) * 100, 2)
acc_perceptron
# Linear SVC 모델
linear_svc = LinearSVC()
linear_svc.fit(train_x, train_y)
Y_pred = linear_svc.predict(test_x)
acc_linear_svc = round(linear_svc.score(train_x, train_y) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent 모델

sgd = SGDClassifier()
sgd.fit(train_x, train_y)
Y_pred = sgd.predict(test_x)
acc_sgd = round(sgd.score(train_x, train_y) * 100, 2)
acc_sgd
# Decision Tree 모델

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_x, train_y)
Y_pred = decision_tree.predict(test_x)
acc_decision_tree = round(decision_tree.score(train_x, train_y) * 100, 2)
acc_decision_tree
# Random Forest 모델

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_x, train_y)
Y_pred = random_forest.predict(test_x)
acc_random_forest = round(random_forest.score(train_x, train_y) * 100, 2)
acc_random_forest
# 머신러닝 클래스 예측 알고리즘에 따른 정확도 평가

models = pd.DataFrame({
    'Model' : ['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC', 'Decision Tree', 'tensorflow_model'],
    'Score' : [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree, acc_nn]
})
models.sort_values(by='Score', ascending=False)
rf_feature_importance = random_forest.feature_importances_
rf_feat_imp = Series(rf_feature_importance, index = feature)

plt.figure(figsize=(10, 10))
rf_feat_imp.sort_values(ascending=True).plot.barh()
plt.xlabel('Feature importance')
plt.ylabel('Feature')
plt.show()
submission = pd.read_csv('../input/gender_submission.csv')
submission.head()
# 예측 결과 csv로 저장

test = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
    "PassengerId":test["PassengerId"],
    "Survived":nn_predicted_result
})
submission.to_csv("./submission.csv", header=True, index=False)

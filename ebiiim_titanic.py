import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import display

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# 読み込み
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
# いくつか表示してみる
print('train_df.head(3)')
display(train_df.head(3))
print('test_df.tail(3)')
display(test_df.tail(3))
# データ全体の情報
train_df.info()
print('-' * 40)
test_df.info()
# データ型が数値の各属性の統計量
train_df.describe()
# データ型がobjectの各属性の統計量
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head(3)
# いらないものをdrop
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head(3)

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
guess_ages = np.zeros((2,3))
print(guess_ages)


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

print(guess_ages)

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

print(train_df.head())

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head(3)

# 不要となったAgeをdropする
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head(3)
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head(3)
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head(3)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head(3)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
display(train_df.head(10))
display(test_df.head(10))
# scikit-learnのフォーマットにあわせる
train_data = train_df.drop("Survived", axis=1)
train_target = train_df["Survived"]
test_data = test_df.drop("PassengerId", axis=1).copy()
train_data.shape, train_target.shape, test_data.shape
# 訓練データセットを3:1で分割する
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, random_state=0)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)  # 訓練

print(knn.score(X_train, y_train))  # 訓練スコア
print(knn.score(X_test, y_test))  # テストスコア
# k値をいろいろ試してみる
def plot_accuracy_k_range(train_data, train_target, k_range=range(1, 21, 2), random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, random_state=random_state)

    (training_accuracy, test_accuracy) = ([], [])
    k_range = k_range

    for k in k_range:
        clf =  KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        # 訓練セット精度を記録
        training_accuracy.append(clf.score(X_train, y_train))
        # 汎化精度を記録
        test_accuracy.append(clf.score(X_test, y_test))

    plt.figure(figsize=(4,3))
    plt.plot(k_range, training_accuracy, label='training accuracy')
    plt.plot(k_range, test_accuracy, label='test accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.legend()

# どうやらk=3がよさげ？
plot_accuracy_k_range(train_data, train_target, random_state=0)
plot_accuracy_k_range(train_data, train_target, k_range=range(1, 668, 3))
# よさげなk値を見つけたら、訓練データセットを丸ごと訓練用にしてpredictする
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_data, train_target)  # 訓練
pred = knn.predict(test_data)
print(pred[0:10])  # 結果の最初の10個
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, random_state=0)

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)

print(logreg.score(X_train, y_train))
print(logreg.score(X_test, y_test))
coeff_df = pd.DataFrame(train_data.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
def plot_logreg_coef(train_data, train_target, random_state=0, c=[10, 0.1, 0.001], penalty='l2', marker=['s', '^', 'v']):
    if len(c) != len(marker):
        print('len(c) != len(marker)'); return
    plt.figure()
    for C, m in zip(c, marker):
        plt.plot(LogisticRegression(C=C, penalty=penalty).fit(train_data, train_target).coef_.T, m, label="C={}".format(C))
    plt.xticks(range(train_data.shape[1]), train_data.columns, rotation=90)
    plt.xlabel('Feature')
    plt.xlabel('Coefficient magnitude')
    plt.hlines(0, 0, train_data.shape[1])
    plt.ylim(-5, 5)
    plt.legend()

plot_logreg_coef(X_train, y_train)
plot_logreg_coef(X_train, y_train, penalty='l1')
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, random_state=0)

linear_svm = LinearSVC(C=1).fit(X_train, y_train)
print(linear_svm.score(X_train, y_train))
print(linear_svm.score(X_test, y_test))
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, random_state=0)

svm = SVC(C=10).fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))
X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, random_state=0)

random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, y_train)
print(random_forest.score(X_train, y_train))
print(random_forest.score(X_test, y_test))
random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(train_data, train_target)
pred = random_forest.predict(test_data)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred
    })

print(submission)
submission.to_csv('./submission.csv', index=False)

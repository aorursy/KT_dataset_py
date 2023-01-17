# 訓練データの読み込み

import pandas as pd, numpy as np

import seaborn as sns, matplotlib.pyplot as plt

import random as rnd



train_df = pd.read_csv("../input/train.csv", header=0)

test_df = pd.read_csv("../input/test.csv", header=0)

dfd = {'train': train_df, 'test': test_df}

for name,df in dfd.items():

    print(name)

    print(df.columns.values)

    print(df.head(3))

    print(df.info())

    print(df.describe(include='all'))
# categoricalな値の範囲

for name,df in dfd.items():

    print(name)

    for column in ['Sex', 'Cabin', 'Embarked']:

        print(df[column].unique())
# 欠損値を確認

for name,df in dfd.items():

    print(name)

    print(df.isnull().sum())
# 欠損値を補完

for name,df in dfd.items():

    print(name)

    for column in ["Age", "Fare"]:

        median_age = df[column].dropna().median()

        if len(df[column][ df[column].isnull() ]) > 0:

            df.loc[(df[column].isnull()), column] = median_age   # とりあえず全体の平均

        print(column, median_age)

    if len(df['Embarked'][ df['Embarked'].isnull() ]) > 0:

        df.loc[(df['Embarked'].isnull()), 'Embarked'] = 'S'   # 最頻値
# categoricalな値とSurvivedとの相関

for column in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:

    print(train_df[[column, 'Survived']].groupby([column], as_index=False).mean().sort_values(by='Survived', ascending=False))
# categoricalな値とSurvivedとの相関

sns.factorplot(data=train_df, x='Pclass', y='Survived', hue='Sex', col='Embarked')

plt.show()
# numericalな値とSurvivedとの相関

for column in ['Age', 'Fare']:

    g = sns.FacetGrid(train_df, col='Survived')

    g.map(plt.hist, column, bins=20)

plt.show()
# categoricalおよびnumericalな値とSurvivedとの相関

# 時間がかかる。。

# for value in ['Age', 'Fare']:

#     for row in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']:

#         g = sns.FacetGrid(train_df, col='Survived', row=row)

#         g.map(plt.hist, value, bins=20)

# plt.show()
for name,df in dfd.items():

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    df['withFamily'] = 1

    df.loc[df['FamilySize']==1, 'withFamily'] = 0

for column in ['FamilySize', 'withFamily']:

    print(train_df[[column, 'Survived']].groupby([column], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Titleの種類

for name,df in dfd.items():

    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)

    print(name)

    print(pd.crosstab(df['Title'], df['Sex']))
# Titleを置換

for name,df in dfd.items():

    df['Title'] = df['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')

    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
# 置換漏れを確認

for name,df in dfd.items():

    print(name)

    print(df['Title'].unique())
# 使用方法を思いつかないデータを削除 (テストデータのIDは別に保存する)

ids = test_df["PassengerId"].values

for name,df in dfd.items():

    df.drop(["Ticket", "PassengerId"], axis=1, inplace=True)

    print(name)

    print(df.head(3))
# ダミー変数に変換

for name,df in dfd.items():

    df["Pclass"] = df["Pclass"].map( {1: 3, 2: 2, 3: 1} ).astype(int)

    df["Sex"] = df["Sex"].map( {"female": 1, "male": 0} ).astype(int)

    df["Embarked"] = df["Embarked"].map( {"Q": 3, "C": 2, "S": 1} ).astype(int)

    df["Title"] = df["Title"].map( {"Mrs": 5, "Miss": 4, "Master": 3, "Rare": 2, "Mr": 1} ).astype(int)

    print(name)

    print(df.head(3))
# 欠損値を補完

mean_ages = np.zeros((2, 3))

for name,df in dfd.items():

    for i in range(0, 2):

        for j in range(0, 3):

            mean_df = df[(df['Sex']==i) & (df['Pclass']==j+1)]['Age'].dropna()

            df.loc[(df['Age'].isnull()) & (df['Sex']==i) & (df['Pclass']==j+1), 'Age'] = int(mean_df.median())

    df['Age'] = df['Age'].astype(int)
sns.pairplot(train_df[['Survived', 'Age', 'Fare']], hue='Survived')

plt.show()
# for name,df in dfd.items():

#     df['Age'] = df['Age'] / 10

#     print(name)

#     print(df.head())

# Banding

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

train_df.drop(["AgeBand"], axis=1, inplace=True)



for name,df in dfd.items():

    df.loc[ df['Age'] <= 16, 'Age'] = 5

    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 4

    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 3

    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 2

    df.loc[ df['Age'] > 64, 'Age'] = 1
# for name,df in dfd.items():

#     df['Fare'] = df['Fare'] / 50

#     print(name)

#     print(df.head())

# Banding

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

train_df.drop(["FareBand"], axis=1, inplace=True)



for name,df in dfd.items():

    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0

    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2

    df.loc[ df['Fare'] > 31, 'Fare'] = 3

    df['Fare'] = df['Fare'].astype(int)

    print(name)

    print(df.head())
for name,df in dfd.items():

    df['Name'] = df['Name'].apply(len)

#train_df['Name'] = pd.qcut(train_df['Name'], 4)

#for name,df in dfd.items():

    df.loc[ df['Name'] <= 20, 'Name'] = 1

    df.loc[(df['Name'] > 20) & (df['Name'] <= 25), 'Name'] = 2

    df.loc[(df['Name'] > 25) & (df['Name'] <= 30), 'Name'] = 3

    df.loc[ df['Name'] > 30, 'Name'] = 4

    df['Name'] = df['Name'].astype(int)

    print(name)

    print(df.head())

print(train_df[['Name', 'Survived']].groupby(['Name'], as_index=False).mean().sort_values(by='Name', ascending=True))
for name,df in dfd.items():

    df['Cabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

    print(name)

    print(df.head())
# categoricalな値とSurvivedとの相関

# for column in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'withFamily', 'FamilySize', 'Age', 'Fare', 'Title']:

#     print(train_df[[column, 'Survived']].groupby([column], as_index=False).mean().sort_values(by='Survived', ascending=False))
# 各feature間の相関

plt.figure(figsize=(12, 8))

sns.heatmap(train_df.astype(float).corr(), cmap=plt.cm.viridis, annot=True)

plt.show()
# 各feature間の相関

# categoricalな値同士だと、点が重なって役に立たない

plt.figure(figsize=(12, 8))

sns.pairplot(train_df, hue='Survived')

plt.show()
for name,df in dfd.items():

    df.drop(["SibSp", "Parch"], axis=1, inplace=True)

    print(name)

    print(df.head(3))
# 各featureとSurvivedとの相関

for column in df:

    print(train_df[[column, 'Survived']].groupby([column], as_index=False).mean().sort_values(by='Survived', ascending=False))
# パラメータ探索

from sklearn import svm

from sklearn.model_selection import GridSearchCV



train_x = train_df.values[0::,1::]

train_y = train_df.values[0::,0]

test_x = test_df.values



C_list = np.logspace(-1, 3, 5)

gamma_list = np.logspace(-2, 0, 3)

tuned_parameters = [

#    {'C': C_list, 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': gamma_list},

#    {'C': C_list, 'kernel': ['sigmoid'], 'gamma': gamma_list},

#    {'C': C_list, 'kernel': ['linear']},

    {'C': C_list, 'kernel': ['rbf'], 'gamma': gamma_list}]

svc = svm.SVC()

clf = GridSearchCV(svc, tuned_parameters, cv=5)

clf.fit(train_x, train_y)

#print(clf.best_estimator_)

print(clf.best_score_, clf.best_params_)

results = clf.cv_results_

for mean, std, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):

    print("{:0.3f} (+/-{:0.03f} for {}".format(mean, std, params))
# 学習

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold, cross_val_score



# Random Forest parameters

rf_params = {

    'n_jobs': -1,

    'n_estimators': 500,

     'warm_start': True, 

     #'max_features': 0.2,

    'max_depth': 6,

    'min_samples_leaf': 2,

    'max_features' : 'sqrt',

    'verbose': 0

}

# Extra Trees Parameters

et_params = {

    'n_jobs': -1,

    'n_estimators':500,

    #'max_features': 0.5,

    'max_depth': 8,

    'min_samples_leaf': 2,

    'verbose': 0

}

# AdaBoost parameters

ada_params = {

    'n_estimators': 500,

    'learning_rate' : 0.75

}

# Gradient Boosting parameters

gb_params = {

    'n_estimators': 500,

     #'max_features': 0.2,

    'max_depth': 5,

    'min_samples_leaf': 2,

    'verbose': 0

}

# Support Vector Classifier parameters 

svc_params = {

    'kernel' : 'rbf',

    'C' : 1,

    'gamma' : 0.1

}



models = []

models.append(('random_forest', RandomForestClassifier(**rf_params)))

models.append(('decision_tree', DecisionTreeClassifier()))

models.append(('extra_trees', ExtraTreesClassifier(**et_params)))

models.append(('ada_boost', AdaBoostClassifier(**ada_params)))

models.append(('gradient_boosting', GradientBoostingClassifier(**gb_params)))

models.append(('perceptron', Perceptron()))

models.append(('sgd_classifier', SGDClassifier()))

models.append(('logistic_regression', LogisticRegression()))

models.append(('svm', SVC(**svc_params)))

models.append(('linear_svc', LinearSVC()))

models.append(('k-nearest_neighbors', KNeighborsClassifier(n_neighbors=3)))

models.append(('gaussian_naive bayes', GaussianNB()))

k_fold = KFold(n_splits=3)

for name,model in models:

    scores = cross_val_score(model, train_x, train_y, cv=k_fold)

    print('{} : mean {:.3f}, std {:.3f}'.format(name, np.mean(scores), np.std(scores)))
# 予測

for name,model in models:

    model.fit(train_x, train_y)

    output = model.predict(test_x).astype(int)

    submit = pd.DataFrame(data={'PassengerId':ids, 'Survived':output})

    submit.to_csv('{}_submit.csv'.format(name), index=False)

    print(name)

    #print(submit.head(3))
# 各featureの寄与度

importances = {}

for name,model in models:

    if name in ['logistic_regression', 'sgd_classifier', 'perceptron', 'svm', 'linear_svc', 'k-nearest_neighbors', 'gaussian_naive bayes']:

       continue

    importances[name] = model.feature_importances_

pd.DataFrame(importances, index=test_df.columns).plot()

plt.show()
# 予測誤りの傾向

from sklearn.metrics import confusion_matrix, classification_report

for name,model in models:

    print(name)

    predict_y = model.predict(train_x)

    print(confusion_matrix(train_y, predict_y))

    print(classification_report(train_y, predict_y))
# 出力したファイルの内容を確認

submit.to_csv('random_forest_submit.csv', index=False)

!head random_forest_submit.csv
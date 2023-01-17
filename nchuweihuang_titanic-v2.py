# !apt-get install build-essential swig

# !pip install --upgrade pip
# !curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

# !pip install auto-sklearn
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

warnings.filterwarnings('ignore')



# Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, neural_network

# from autosklearn import classification

from xgboost import XGBClassifier



# Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



dataset = train.append(test)

dataset.describe(include='all')
print(train.isnull().sum())
print(test.isnull().sum())
# 針對 Age、Embarked、Fare 進行遺漏值填補

# train['Age'].fillna(dataset['Age'].median(), inplace=True)

# test['Age'].fillna(dataset['Age'].median(), inplace=True)



train['Embarked'].fillna(dataset['Embarked'].mode(), inplace=True)

test['Embarked'].fillna(dataset['Embarked'].mode(), inplace=True)



test['Fare'].fillna(dataset['Fare'].mode(), inplace=True)



# 擴增特徵 稱謂、家庭人口

dataset['Title'] = dataset['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]

train['Title'] = train['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]

test['Title'] = test['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]



print(dataset["Title"].value_counts())



# 將稱謂數量低於 10 分類為 Misc

train['Title'] = train['Title'].apply(lambda x: 'Misc' \

                                       if (dataset['Title'].value_counts() < 5)[x] == True \

                                       else x)

test['Title'] = test['Title'].apply(lambda x: 'Misc' \

                                       if (dataset['Title'].value_counts() < 5)[x] == True \

                                       else x)



train['Family'] = train['SibSp'] + train['Parch'] + 1

test['Family'] = test['SibSp'] + test['Parch'] + 1
train['Family'].value_counts()
sns.boxplot(y='Pclass', x=np.log10(dataset['Fare']), hue='Survived', data=dataset, orient='h', palette='Set3')
# 票價分布非常廣且傾斜，價差非常大，因此取log使區間所小一些。

train['Fare'] = (train['Fare'] + 1).map(lambda x: np.log10(x) if x > 0 else 0)

test['Fare'] = (test['Fare'] + 1).map(lambda x: np.log10(x) if x > 0 else 0)



label = LabelEncoder()



# 切分成 4, 5, 6 不同區間，測試票價

train['Fare_4'] = label.fit_transform(pd.qcut(train['Fare'], 4))

train['Fare_5'] = label.fit_transform(pd.qcut(train['Fare'], 5))

train['Fare_6'] = label.fit_transform(pd.qcut(train['Fare'], 6))



test['Fare_4'] = label.fit_transform(pd.qcut(test['Fare'], 4))

test['Fare_5'] = label.fit_transform(pd.qcut(test['Fare'], 5))

test['Fare_6'] = label.fit_transform(pd.qcut(test['Fare'], 6))
train['Connected'] = 0.5

for _, group in train.groupby('Ticket'):

    if (len(group) > 1):

        for index, row in group.iterrows():

            smax = group.drop(index)['Survived'].max()

            smin = group.drop(index)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                train.loc[train['PassengerId'] == passID, 'Connected'] = 1

            elif (smin == 0.0):

                train.loc[train['PassengerId'] == passID, 'Connected'] = 0

train.groupby('Connected')[['Survived']].mean().round(3)
test['Connected'] = 0.5
train['Title'] = train['Title'].str.replace(" ", "")

test['Title'] = test['Title'].str.replace(" ", "")
train['Title'] = train['Title'].map({"Dr": 0, "Master": 1, "Miss": 2, "Mr": 3, "Mrs": 4, "Rev": 5, "Misc": 6})

train.groupby('Title')['Age'].median()
test['Title'] = test['Title'].map({"Dr": 0, "Master": 1, "Miss": 2, "Mr": 3, "Mrs": 4, "Rev": 5, "Misc": 6})

test.groupby('Title')['Age'].median()
age = train.groupby('Title')['Age'].median().values

train['Ti_Age'] = train['Age']

for i in range(0, 7):

    train.loc[(train.Age.isnull()) & (train.Title == i), 'Ti_Age'] = age[i]



train['Ti_Age'] = train['Ti_Age'].astype('int')

train['Ti_Age'] = ((train['Ti_Age']) < 16.0) * 1



age = test.groupby('Title')['Age'].median().values

test['Ti_Age'] = test['Age']

for i in range(0, 7):

    test.loc[(test.Age.isnull()) & (test.Title == i), 'Ti_Age'] = age[i]



test['Ti_Age'] = test['Ti_Age'].astype('int')

test['Ti_Age'] = ((test['Ti_Age']) < 16.0) * 1
# label-encoding

# train['Sex'] = label.fit_transform(train['Sex'])

# test['Sex'] = label.fit_transform(test['Sex'])



# train['Embarked'] = label.fit_transform(train['Embarked'].astype(str))

# test['Embarked'] = label.fit_transform(test['Embarked'].astype(str))



# train['Title'] = label.fit_transform(train['Title'])

# test['Title'] = label.fit_transform(test['Title'])



x_PS = ['Pclass', 'Sex', 'Connected']

X_AFE = ['Age', 'Fare', 'Embarked']

x_new = ['Title', 'Family']



x_fare_4, x_fare_5, x_fare_6  = ['Fare_4'], ['Fare_5'], ['Fare_6']



Target = ['Survived']



# One-hot encoding

train = pd.get_dummies(train[Target + x_PS + x_fare_4 + x_fare_5 + x_fare_6])

test = pd.get_dummies(test[x_PS + x_fare_4 + x_fare_5 + x_fare_6])

train.head()
# 使用前向選擇法做特徵選取 (RFE)

selector = feature_selection.RFECV(ensemble.RandomForestClassifier(n_estimators=250, min_samples_split=20), cv=10, n_jobs=-1)

selector.fit(train, train['Survived'])

print(selector.support_)

print(selector.ranking_)

print(selector.grid_scores_*100)
score_b4, score_b5, score_b6 = [], [], []

seeds = 10

for i in range(seeds):

    diff_cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=i)

    selector = feature_selection.RFECV(ensemble.RandomForestClassifier(random_state=i,

                                                                       n_estimators=250,

                                                                       min_samples_split=20),

                                       cv=diff_cv, n_jobs=-1)

    selector.fit(train, train['Survived'])

    score_b4.append(selector.grid_scores_[2])

    score_b5.append(selector.grid_scores_[3])

    score_b6.append(selector.grid_scores_[4])
score_list = [score_b4, score_b5, score_b6]

for item in score_list:

    item = np.array(item*100)

# plot

fig = plt.figure(figsize= (18,8))

ax = plt.gca()

ax.plot(range(seeds), score_b4,'-ok',label='bins = 4')

ax.plot(range(seeds), score_b5,'-og',label='bins = 5')

ax.plot(range(seeds), score_b6,'-ob',label='bins = 6')

ax.set_xlabel("Seed #", fontsize = '14')

ax.set_ylim(0.783,0.815)

ax.set_ylabel("Accuracy", fontsize = '14')

ax.set_title('bins = 4 vs bins = 5 vs bins = 6', fontsize='20')

plt.legend(fontsize = 14,loc='upper right')
print(train.isnull().sum())

print(test.isnull().sum())
# 切割訓練資料，將20%作為驗證資料，80%作為訓練資料

x_train, x_test, y_train, y_test = model_selection.train_test_split(train, train[Target], random_state = 0)

print('Original Train Shape: {}'.format(train.shape))

print('Target Shape: {}'.format(train[Target].shape))

print('Train Shape: {}'.format(np.asarray(x_train).shape))

print('Test Shape: {}'.format(np.asarray(x_test).shape))
def predictor(dropout, csvname):

    # run model 10x with 60/30 split intentionally leaving out 10%

    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

    MLA = [

        ensemble.AdaBoostClassifier(),

        ensemble.BaggingClassifier(),

        ensemble.GradientBoostingClassifier(),

        ensemble.RandomForestClassifier(random_state=2, n_estimators=250, min_samples_split=20,),



        gaussian_process.GaussianProcessClassifier(),



        linear_model.LogisticRegressionCV(),

        linear_model.SGDClassifier(),

        linear_model.Perceptron(),



        naive_bayes.GaussianNB(),



        neighbors.KNeighborsClassifier(),



        svm.SVC(probability=True),

        svm.LinearSVC(),



        tree.DecisionTreeClassifier(),

        

        neural_network.MLPClassifier(),

        

        XGBClassifier()

        

#         classification.AutoSklearnClassifier()

    ]



    MLA_columns = ['Name', 'Parameters','Train Accuracy Mean', 'Test Accuracy Mean', 'Test Accuracy 3*STD' ,'Time']

    MLA_compare = pd.DataFrame(columns = MLA_columns)

    MLA_predict = train[Target]



    row_index = 0

    for alg in MLA:

        MLA_name = alg.__class__.__name__

        MLA_compare.loc[row_index, 'Name'] = MLA_name

        MLA_compare.loc[row_index, 'Parameters'] = str(alg.get_params())



        cv_results = model_selection.cross_validate(alg, train.drop(columns=dropout+Target), train[Target], cv=cv_split)

        MLA_compare.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

        MLA_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()

        MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()

        MLA_compare.loc[row_index, 'Test Accuracy 3*STD'] = cv_results['train_score'].std()*3



        alg.fit(train.drop(columns=dropout+Target), train[Target])

        MLA_predict[MLA_name] = alg.predict(train.drop(columns=dropout+Target))



        row_index += 1



    MLA_compare.sort_values(by=['Test Accuracy Mean'], ascending=False, inplace=True)

    

    # Submission File

    y_hat = MLA[MLA_compare.index[0]].predict(test.drop(columns=dropout))

    output = pd.DataFrame({"PassengerId": pd.read_csv('../input/test.csv')['PassengerId'],

                      "Survived": y_hat})

    output.to_csv(csvname + '.csv', index=False)

    

    # emsemble

    eclf_hard = ensemble.VotingClassifier(estimators = [('GradientBoostingClassifier', MLA[2]), 

                                               ('XGBClassifier', MLA[14]), 

                                               ('SVC', MLA[10]),

                                               ('GaussianProcessClassifier', MLA[4]),

                                               ('RandomForestClassifier', MLA[3]),

                                               ('BaggingClassifier', MLA[1]), 

                                               ('DecisionTreeClassifier', MLA[12]),

                                               ('KNeighborsClassifier', MLA[9]),

                                               ('LinerSVC', MLA[11]),

                                               ('LogisticRegressionCV', MLA[5]),

                                               ('AdaBoostClassifier', MLA[0]),

                                               ('GaussianNB', MLA[8]),

                                               ('SGDClassifier', MLA[6]),

                                               ('Perceptron', MLA[7])], voting='hard',

                                                                        weights=[14,

                                                                                 13,

                                                                                 12,

                                                                                 11,

                                                                                 10,

                                                                                 9,

                                                                                 8,

                                                                                  7,

                                                                                 6, 

                                                                                  5, 

                                                                                  4, 

                                                                                  3, 

                                                                                  2, 

                                                                                  1])

    eclf_hard.fit(train.drop(columns=dropout+Target), train[Target])

    cv_results = model_selection.cross_validate(eclf_hard, train.drop(columns=dropout+Target), train[Target], cv=cv_split)

    MLA_compare.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()

    MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()

    MLA_compare.loc[row_index, 'Test Accuracy 3*STD'] = cv_results['train_score'].std()*3

    

    # submit emsemble

    y_hat = eclf_hard.predict(test.drop(columns=dropout))

    output = pd.DataFrame({"PassengerId": pd.read_csv('../input/test.csv')['PassengerId'],

                      "Survived": y_hat})

    output.to_csv('emsemble_' + csvname + '.csv', index=False)

    

    # auto-sklearn

#     y_hat = MLA[15].predict(test.drop(columns=dropout))

#     output = pd.DataFrame({"PassengerId": pd.read_csv('../input/test.csv')['PassengerId'],

#                       "Survived": y_hat})

#     output.to_csv('autosklearn_' + csvname + '.csv', index=False)

    

    return MLA_compare
predictor(['Fare_4', 'Fare_5'], 'fare_4')
predictor(['Fare_4', 'Fare_6'], 'fare_5')
predictor(['Fare_4', 'Fare_5'], 'fare_6')
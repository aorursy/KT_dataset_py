# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from statsmodels.graphics.mosaicplot import mosaic

import seaborn as sb



from sklearn import linear_model, metrics, preprocessing, model_selection

from sklearn.model_selection import KFold



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("../input/titanic/train.csv", delimiter=",")



train_data.head(5)
train_data = pd.read_csv("../input/titanic/train.csv", delimiter=",")



engineer_data = np.c_[train_data['Sex'], train_data['Survived']]



myDataframe = pd.DataFrame(engineer_data, columns=['Sex', 'Survived'])



mosaic(data=myDataframe, index=['Sex', 'Survived'], title='Mosaic Plot')



plt.show()
engineer_data = np.c_[train_data['Pclass'], train_data['Survived']]



myDataframe = pd.DataFrame(engineer_data, columns=['Pclass', 'Survived'])



mosaic(data=myDataframe, index=['Pclass', 'Survived'], title='Mosaic Plot')



plt.show()

engineer_data = np.c_[train_data['Embarked'], train_data['Survived']]



myDataframe = pd.DataFrame(engineer_data, columns=['Embarked', 'Survived'])



mosaic(data=myDataframe, index=['Embarked', 'Survived'], title='Mosaic Plot')



plt.show()



survived_train_data1 = train_data[train_data.Survived == 1]['Fare']

survived_train_data2 = train_data[train_data.Survived == 0]['Fare']



data1 = survived_train_data1.dropna(how='all').values.tolist()

data2 = survived_train_data2.dropna(how='all').values.tolist()



data1 = list(filter(lambda x: x != 0, data1))

data2 = list(filter(lambda x: x != 0, data2))



data1 = np.log(data1)

data2 = np.log(data2)



beard = (data1, data2)



fig = plt.figure()



ax = fig.subplots()



bp = ax.boxplot(beard)



ax.set_xticklabels(['Survived', 'Dead'])



plt.title('Box plot')



plt.grid()



plt.xlabel('Survived')



plt.ylabel('Fare')



plt.ylim([0,8])

plt.show()
survived_train_data1 = train_data[train_data.Survived == 1]['SibSp']

survived_train_data2 = train_data[train_data.Survived == 0]['SibSp']



data1 = survived_train_data1.dropna(how='all').values.tolist()

data2 = survived_train_data2.dropna(how='all').values.tolist()



data1 = list(filter(lambda x: x != 0, data1))

data2 = list(filter(lambda x: x != 0, data2))



hige = (data1, data2)



fig = plt.figure()



ax = fig.subplots()



bp = ax.boxplot(hige)



ax.set_xticklabels(['Survived', 'Dead'])



plt.title('Box plot')



plt.grid()



plt.xlabel('Survived')



plt.ylabel('SibSp')



plt.ylim([0,10])



plt.show()



train_data = pd.read_csv("../input/titanic/train.csv", delimiter=",")



print('【データ総数】\n', train_data.shape[0])



print('【欠損データ有無】\n', train_data.isnull().any())



print('【欠損データ数】\n', train_data.isnull().sum())



print('【年齢:平均値】', train_data['Age'].dropna().mean())

print('【年齢:中央値】', train_data['Age'].dropna().median())

print('【年齢:標準偏差】', train_data['Age'].dropna().std())

print('【年齢：範囲】', train_data['Age'].dropna().min(), '~', train_data['Age'].dropna().max())

train_data['Age'].plot(kind='hist', bins=50, subplots=True);

plt.show()



train_data = train_data.dropna(subset=['Embarked'])





train_data_cabin = train_data['Cabin']

train_data = train_data.drop(['Cabin'], axis=1)

train_data = train_data.fillna(train_data.median()['Age'])

train_data['Cabin'] = train_data_cabin



print('【欠損データ有無】\n', train_data.isnull().any())

print('【データ総数】\n', train_data.shape[0])



train_data = train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

train_data.head(10)



train_data['male'] = (train_data['Sex'] == 'male').astype(int)

train_data['female'] = (train_data['Sex'] == 'female').astype(int)



train_data['embarked_c'] = (train_data['Embarked'] == 'C').astype(int)

train_data['embarked_q'] = (train_data['Embarked'] == 'Q').astype(int)

train_data['embarked_s'] = (train_data['Embarked'] == 'S').astype(int)



train_data = train_data.drop(['Sex', 'Embarked'], axis=1)



train_data.to_csv('train2.csv')

train_data.head(10)



source_train_data = pd.read_csv("train2.csv", delimiter=",")



source_train_data_x = source_train_data.loc[:, 'Pclass': 'embarked_s'].values

source_train_data_y = source_train_data.loc[:, 'Survived'].values



sc=preprocessing.StandardScaler()

sc.fit(source_train_data_x)



source_train_data_x_std=sc.transform(source_train_data_x)
array_loss = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]



array_iter = [5, 10, 50, 100, 150, 200, 250, 300, 400, 500]



score_k = 10

array_scores = np.empty((len(array_loss), len(array_iter), score_k))



for i, loss in enumerate(array_loss):

    for j, iter_cnt in enumerate(array_iter):

        clf_result=linear_model.SGDClassifier(loss=loss, max_iter=iter_cnt, tol=None)

        kf = KFold(n_splits=score_k, shuffle=False, random_state=None)

        scores=model_selection.cross_val_score(clf_result, source_train_data_x_std, source_train_data_y, cv=kf)

        array_scores[i][j] = scores
fig, ax = plt.subplots(nrows=1, ncols=len(array_loss), figsize=(20, 8))

for i, loss in enumerate(array_loss):

    array_means = np.empty(len(array_iter))

    for j, iter_cnt in enumerate(array_iter):

        array_means[j] = array_scores[i][j].mean()

    ax[i].plot(array_iter, array_means, marker='o')

    ax[i].set_title(loss)

    ax[i].set_xlabel("max_iter")

    ax[i].set_ylabel("mean")
clf_result=linear_model.SGDClassifier(loss="hinge", max_iter=200, tol=None)
scores=model_selection.cross_val_score(clf_result, source_train_data_x_std, source_train_data_y, cv=kf)

print(scores)

print("平均正解率 = ", scores.mean())

print("正解率の標準偏差 = ", scores.std())
X_train, X_test, train_label, test_label=model_selection.train_test_split(source_train_data_x_std,source_train_data_y, test_size=0.3, random_state=0)



clf_result.fit(X_train, train_label)



pre=clf_result.predict(X_test)



ac_score=metrics.accuracy_score(test_label,pre)



print("正答率 = ",ac_score)
train_data = pd.read_csv("../input/titanic/test.csv", delimiter=",")



print('【データ総数】\n', train_data.shape[0])

print('【欠損データ有無】\n', train_data.isnull().any())

print('【欠損データ数】\n', train_data.isnull().sum())

print('【年齢:平均値】', train_data['Fare'].dropna().mean())

print('【年齢:中央値】', train_data['Fare'].dropna().median())

print('【年齢:標準偏差】', train_data['Fare'].dropna().std())

print('【年齢：範囲】', train_data['Fare'].dropna().min(), '~', train_data['Fare'].dropna().max())

train_data['Fare'].plot(kind='hist', bins=100, subplots=True);

plt.show()



tmp_fare_data = train_data[train_data['Fare'].isnull()]

train_data = train_data.dropna(subset=['Fare'])

tmp_fare_data['Fare'] = train_data.median()['Fare']

train_data = train_data.append(tmp_fare_data)

train_data_cabin = train_data['Cabin']

train_data = train_data.drop(['Cabin'], axis=1)

train_data = train_data.fillna(train_data.median()['Age'])

train_data['Cabin'] = train_data_cabin



print('【欠損データ有無】\n', train_data.isnull().any())



train_data = train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

train_data.head(10)



train_data['male'] = (train_data['Sex'] == 'male').astype(int)

train_data['female'] = (train_data['Sex'] == 'female').astype(int)



train_data['embarked_c'] = (train_data['Embarked'] == 'C').astype(int)

train_data['embarked_q'] = (train_data['Embarked'] == 'Q').astype(int)

train_data['embarked_s'] = (train_data['Embarked'] == 'S').astype(int)



train_data = train_data.drop(['Sex', 'Embarked'], axis=1)



train_data.to_csv('test2.csv')

train_data.head(10)



tmp_passengerid = train_data['PassengerId']

train_data = train_data.loc[:, 'Pclass': 'embarked_s'].values



sc=preprocessing.StandardScaler()

sc.fit(train_data)



source_train_data_x_std=sc.transform(train_data)



pre=clf_result.predict(source_train_data_x_std)

submission_data = pd.DataFrame()

submission_data['PassengerId'] = tmp_passengerid

submission_data['Survived'] =pre

submission_data.to_csv('submission.csv', index=False)
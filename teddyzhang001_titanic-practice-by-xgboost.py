# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def load_data(file_name, is_train):
    data = pd.read_csv(file_name)  # 数据文件路径
    pd.set_option('display.width', 200)
    print('data.describe() = \n', data.describe())

    # 性别
    # data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['Sex'] = pd.Categorical(data['Sex']).codes

    # 补齐船票价格缺失值
    if len(data.Fare[data.Fare == 0]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data[data['Pclass'] == f + 1]['Fare'].dropna().median()
        print(fare)
        for f in range(0, 3):  # loop 0 to 2
            data.loc[(data.Fare == 0) & (data.Pclass == f + 1), 'Fare'] = fare[f]

    print('data.describe() = \n', data.describe())
    # 年龄：使用均值代替缺失值
    # mean_age = data['Age'].dropna().mean()
    # data.loc[(data.Age.isnull()), 'Age'] = mean_age
    if is_train:
        # 年龄：使用随机森林预测年龄缺失值
        print('随机森林预测缺失年龄：--start--')
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]   # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        print(age_exist)
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=20)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print('随机森林预测缺失年龄：--over--')
    else:
        print('随机森林预测缺失年龄2：--start--')
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        # print age_exist
        x = age_exist.values[:, 1:]
        for i in range(332):
            for j in range(4):
                if np.isnan(x[i][j]):
                    x[i][j] = 0
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print('随机森林预测缺失年龄2：--over--')
    data['Age'] = pd.cut(data['Age'], bins=6, labels=np.arange(6))

    # 起始城市
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'  # 保留缺失出发城市
    embarked_data = pd.get_dummies(data.Embarked)
    print('embarked_data = ', embarked_data)
    # embarked_data = embarked_data.rename(columns={'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', 'U': 'UnknownCity'})
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)

    print(data.describe())
    data.to_csv('New_Data.csv')

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    # x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = None
    if 'Survived' in data:
        y = data['Survived']

    x = np.array(x)
    y = np.array(y)

    # 思考：这样做，其实发生了什么？
    x = np.tile(x, (5, 1))
    y = np.tile(y, (5, ))
    if is_train:
        return x, y
    return x, data['PassengerId']

x, y = load_data('../input/train.csv', True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print('%s正确率：%.3f%%' % (tip, acc_rate))
    return acc_rate

def write_result(c, c_type):
    file_name = '../input/test.csv'
    x, passenger_id = load_data(file_name, False)

    if c_type == 3:
        x = xgb.DMatrix(x)
    y = c.predict(x)
    y[y > 0.5] = 1
    y[~(y > 0.5)] = 0

    predictions_file = open("Prediction_%d.csv" % c_type, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", "Survived"])
    open_file_object.writerows(list(zip(passenger_id, y)))
    predictions_file.close()

data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
         # 'subsample': 1, 'alpha': 0, 'lambda': 0, 'min_child_weight': 1}
bst = xgb.train(param, data_train, num_boost_round=20, evals=watch_list)
y_hat = bst.predict(data_test)
write_result(bst, 3)
y_hat[y_hat > 0.5] = 1
y_hat[~(y_hat > 0.5)] = 0
xgb_acc = accuracy_score(y_test, y_hat)
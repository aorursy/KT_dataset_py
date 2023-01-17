# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import string as st
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

train= pd.read_csv("../input/train.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
test= pd.read_csv("../input/test.csv").replace("male",0).replace("female",1).replace("S",0).replace("C",1).replace("Q",2)
train.head()
def convert(data):
    train = data.copy()
    # Cabinの頭文字を、それぞれ数値に変換する。null は0に変更
    train['tmpCabin'] = train.Cabin.fillna('U').str.extract('(.)', expand=False)
    train['tmpCabin'] = train.tmpCabin.apply(lambda x: int(x.translate(x.maketrans('ABCDEFGTU', '123456780'))))
    # 年齢を大きく分ける
    train['tmpAge'] = pd.cut(train.Age.fillna(train.Age.mean()), bins=[0,10,20,30,40,100], labels=[10,20,30,40,100]).astype(np.int64)
    # 不要な行削除
    train['Fare'] = train.Fare.fillna(train.Fare.mean())
    train = train.drop(columns=['Cabin','Age','Parch','Ticket','Name', 'Embarked'])
    return train
copy = convert(train)
copy_values = copy.values
xs  = copy_values[:, 2:]
y = copy_values[:, 1]
copy.info()
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV


# parameters = {
#         'n_estimators'      : [10,25,50,75,100],
#         'random_state'      : [0],
#         'n_jobs'            : [4],
#         'min_samples_split' : [5,10, 15, 20,25, 30],
#         'max_depth'         : [5, 10, 15,20,25,30]
# }

# clf = grid_search.GridSearchCV(RandomForestClassifier(), parameters)
# clf.fit(xs, y)
 
# print(clf.best_estimator_)
random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

test_copy  = convert(test)
xs_test = test_copy.values[:, 1:]
random_forest.fit(xs, y)
test_copy.isnull().sum()
y_predict = random_forest.predict(xs_test)
y_predict.size
import csv
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["PassengerId", "Survived"])
    for pid, survived in zip(test_copy.values[:,0].astype(int), y_predict.astype(int)):
        writer.writerow([pid, survived])
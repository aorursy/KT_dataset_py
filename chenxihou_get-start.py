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
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
def get_call_age_dict(train):
    df = train[["Name", "Age"]]
    df["call"] = df["Name"].apply(find_call)
    tmp = df.groupby("call")["Age"].mean().astype(int).to_dict()
    
    return tmp
    

def deal_category_class(df, classes):
    for each in classes:
        tmp = pd.get_dummies(df[each], prefix=each)
        df = pd.concat([df, tmp], axis=1)
        df = df.drop(labels=[each], axis=1)
    return df


def find_call(name):
    tmp = re.findall(r"Mrs|Mr|Miss", name)
    ret = tmp[0] if len(tmp)>0 else 'unknow'
    
    return ret

def fill_age(x):
    age = x["Age"]
    if np.isnan(age):
        age = train[train["call"]==x["call"]]["Age"].mean()
    if np.isnan(age):
        age = train["Age"].mean()
    return age

def fill_cabin(x):
    Cabin = x["Cabin"]
    ret = 1 if isinstance(Cabin, str) else 0
    
    return ret

# train.loc[train["Age"].isnull(), "Age"] = a

train = pd.read_csv("../input/train.csv", sep=',')
test = pd.read_csv("../input/test.csv", sep=',')

call_age_dict = get_call_age_dict(train)

# for data in [train, test]:
#     data["call"] = data["Name"].apply(find_call)
#     data.loc[data["Age"].isnull(), "Age"] = data[pd.isnull(data["Age"])]["call"].map(call_age_dict).values
#     data = deal_category_class(data, ["Sex", "Pclass", "Embarked", "call"])
#     drop_columns = ["Name", "Cabin", "Ticket"]
#     data = data.drop(labels=drop_columns, axis=1)

# deal train
train["call"] = train["Name"].apply(find_call)
train.loc[train["Age"].isnull(), "Age"] = train[pd.isnull(train["Age"])]["call"].map(call_age_dict).values
train = deal_category_class(train, ["Sex", "Pclass", "Embarked", "call"])
drop_columns = ["Name", "Cabin", "Ticket"]
train = train.drop(labels=drop_columns, axis=1)

# deal test
test["call"] = test["Name"].apply(find_call)
test.loc[test["Age"].isnull(), "Age"] = test[pd.isnull(test["Age"])]["call"].map(call_age_dict).values
test = deal_category_class(test, ["Sex", "Pclass", "Embarked", "call"])
drop_columns = ["Name", "Cabin", "Ticket"]
test = test.drop(labels=drop_columns, axis=1)
features = train.drop(labels=["PassengerId", "Survived"], axis=1).values
labels = train["Survived"]
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
para = {
    'booster': 'gbtree',
    'objective': "binary:logistic",
    'subsample': 0.7,
    'silent': 1,
}

kf = KFold(n_splits=5, random_state=0)
result = []
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clf = xgb.XGBClassifier(n_estimators=100, objective="binary:logistic", booster='gbtree', learning_rate=0.1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    result.append(np.mean(y_pred == y_test))

print("accuracy in valid mean::%s, std::%s" % (np.array(result).mean(), np.array(result).std()))
test_features = test.drop(labels="PassengerId", axis=1).values
pre = clf.predict(test_features)
result = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pre})
# result.to_csv("../input/result.csv", index=False, header=True)
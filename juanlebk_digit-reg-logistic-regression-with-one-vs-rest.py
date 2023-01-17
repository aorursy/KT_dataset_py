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
# load data
train_path = '../input/train.csv'
train_data = pd.read_csv(train_path)
train_data_0 = train_data[train_data.iloc[:,0]==0].iloc[:, 1:]
train_data_diff0 = train_data[train_data.iloc[:,0]!=0].iloc[:, 1:]
train_data_0.head()
train_data_diff0.head()
train_data_0.shape
train_data_diff0.shape
ones = np.ones(train_data_0.shape[0])
ones
zeros = np.zeros(train_data_diff0.shape[0])
print(ones.shape)
print(zeros.shape)
y = np.concatenate((np.ones(train_data_0.shape[0]), np.zeros(train_data_diff0.shape[0])), axis=0)
y
X = np.concatenate((train_data_0, train_data_diff0), axis=0)
X.shape
X
# define function to get each digit from data.
import pandas as pd
def get_data (digit = 0, data=None):
    if type(data) is not pd.DataFrame:
        return None
    X_digit_data = data[data.iloc[:,0] == digit].iloc[:,1:]
    X = np.concatenate((X_digit_data, data[data.iloc[:,0] != digit].iloc[:,1:]), axis=0)
    y = np.concatenate((np.ones(X_digit_data.shape[0]), np.zeros(data.shape[0]-X_digit_data.shape[0])), axis=0)
    return X, y
val_data = train_data.sample(frac=0.1, random_state=1).reset_index(drop=True)
val_data.label.value_counts()
train_data.label.value_counts()
# Using train_test_split of module sklearn.model_selection
from sklearn.model_selection import train_test_split
train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=1)
type(train_set)
train_set.label.value_counts()
val_set.label.value_counts()
X_0, y_0 = get_data(digit=0, data=train_set)
from sklearn.linear_model import LogisticRegression

model_0 = LogisticRegression()
model_0.fit(X=X_0, y=y_0)
model_0.coef_
model_0.intercept_
X_0_val, y_0_val = get_data(digit=0, data=val_set)
y_0_pred = model_0.predict(X_0_val)
y_0_proba = model_0.predict_proba(X=X_0_val)
y_0_proba[:,1]
y_0_pred
from sklearn.metrics import accuracy_score, f1_score
accuracy_score(y_true=y_0_val, y_pred=y_0_pred)
f1_score(y_true=y_0_val, y_pred=y_0_pred)
# At here, I practice for class programming in python for fun. They are not more meanings.
# I only think, I should practice more and more about oop. Apply oop in my project Machine Learning.
from sklearn.metrics import accuracy_score, f1_score
class DigitRegression:
    def __init__(self, list_models):
        self.models = list_models
        pass
    def probalities(self, X_data):
        return np.array([model.predict_proba(X=X_data)[:,1] for model in self.models])
    def predict(self, X_data):
        probas = self.probalities(X_data)
        print(probas)
        return np.argmax(probas, axis=0)
    pass
# If have more time, I will use oop in this problem later.
# Prepare all data and get model from them
# list_models = []
# for i in range(10):
#     X_train, y_train = get_data(digit=i, data=train_set)
#     my_model = LogisticRegression()
#     my_model.fit(X=X_train, y=y_train)
#     list_models.append(my_model)
#     pass
# Validation
# dg = DigitRegression(list_models=list_models)
# y_val = val_set.label.values
# X_val = val_set.iloc[:, 1:].values
# my_preds = dg.predict(X_data=X_val)
# my_preds
list_models = []
for i in range(10):
    X_train, y_train = get_data(digit=i, data=train_data)
    my_model = LogisticRegression()
    my_model.fit(X=X_train, y=y_train)
    list_models.append(my_model)
    pass
# load data test
test_path = '../input/test.csv'
test_data = pd.read_csv(test_path)
# Test
dg_test = DigitRegression(list_models=list_models)
# y_test = val_set.label.values
X_test = test_data.values
test_preds = dg_test.predict(X_data=X_test)
print(test_preds)
output = pd.DataFrame({'ImageId': test_data.index + 1,
                       'label': test_preds})

output.to_csv('submission.csv', index=False)
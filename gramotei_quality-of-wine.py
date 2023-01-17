# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
path = '../input/winequality-red.csv'
data = pd.read_csv(path)
Y = data.quality
X = data.iloc[:, :11]

# Preprocessing data
points = (2, 6.5, 8)
group_names = ['bad wine', 'good wine']
data['quality'] = pd.cut(data['quality'], bins=points, labels=group_names)

from sklearn.preprocessing import LabelEncoder, StandardScaler
label = LabelEncoder()
data['quality'] = label.fit_transform(data['quality'])
    
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, random_state=42)

scale = StandardScaler()
train_X = scale.fit_transform(train_X)
test_X = scale.fit_transform(test_X)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


def test_models(amount_leafes):
    model_2 = RandomForestClassifier(max_depth=amount_leafes, n_estimators=50)
    model_2.fit(train_X, train_Y)
    return model_2, cross_val_score(model_2, train_X, train_Y)

best_score = 0
score = 0
opt_amount_leafes = 0
for i in [27, 28, 30, 31, 32, 33]:
    model, score = test_models(i)
    if max(score) > best_score:
        best_score = max(score)
        final_model = model
    print(score)
pred_val = final_model.predict(test_X)

print('Algorithm is RandomForest')
print(classification_report(test_Y, pred_val))


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


def test_models_xg_boost(amount_leafes):
    model_1 = XGBClassifier(amount_leafes)
    model_1.fit(train_X, train_Y)
    return model_1, cross_val_score(model, train_X, train_Y)

best_score = 0
score = 0
for i in [27, 28, 30, 31, 32, 33]:
    model, score = test_models_xg_boost(i)
    if max(score) > best_score:
        best_score = max(score)
        final_model = model

predicted_value = final_model.predict(test_X)
print('Algorithm is XGBClassifier')
print(classification_report(test_Y, predicted_value))

my_submission = pd.DataFrame({'Quality': predicted_value})
my_submission.to_csv('my_submission', index=False)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
train_file_path = '../input/train.csv'
test_file_path = '../input/test.csv'
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)
select_col = ['Pclass', 'Age', 'SibSp', 'Parch']
x_train = train_data[select_col]
x_test = test_data [select_col]
y_train = train_data.Survived
my_imputer = Imputer()
x_train = my_imputer.fit_transform(x_train)
x_test = my_imputer.fit_transform(x_test)
model = XGBRegressor()
model.fit(x_train, y_train)
surv_predict = model.predict(x_test)
print(surv_predict)
n = len(surv_predict)
for i in range(0,n):
    if surv_predict[i]<=0.3:
        surv_predict[i] = 0
    else:
        surv_predict[i] = 1
my_submission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': surv_predict})
print(my_submission)
my_submission.to_csv('my submission.csv', index = False)

# Any results you write to the current directory are saved as output.

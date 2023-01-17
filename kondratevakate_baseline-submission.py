# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score # метрики качества
%matplotlib inline

import matplotlib.pyplot as plt
training_data = pd.read_csv('/kaggle/input/predict-employee-quiting/train_data.csv')
training_data = training_data[training_data.columns[1:]]
training_data.describe().T
training_data.info()
train_mean = training_data.mean()

train_mean
training_data.fillna(train_mean, inplace=True)
target_variable_name = 'Attrition'
training_values = training_data[target_variable_name]
training_points = training_data.drop(target_variable_name, axis=1)
training_points.shape
test_data = pd.read_csv('/kaggle/input/predict-employee-quiting/test_data.csv')
test_data = test_data.drop('Unnamed: 0', axis = 1)
test_data.describe().T
test_data.fillna(train_mean, inplace=True)
id_variable_name = 'index'

ids = test_data[id_variable_name] # записываем столбец id в отдельную переменную

test_points = test_data.drop(id_variable_name, axis=1) # удаляем его из тестовой выборки 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
test_points
text_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
label_encoder = LabelEncoder()
for col in text_features:

    training_points[col] = label_encoder.fit_transform(training_points[col]) + 1

    test_points[col] = label_encoder.transform(test_points[col]) + 1
import xgboost as xgb
xgboost_model = xgb.XGBClassifier(n_estimators=100)
xgboost_model.fit(training_points, training_values)
from sklearn.model_selection import cross_val_predict
cross_validation_predictions = cross_val_predict(xgboost_model, training_points, training_values, 

                                                 cv=5, method='predict_proba')

cross_validation_predictions = cross_validation_predictions[:, 1] # нам нужны вероятности для второго класса
cross_validation_predictions
from sklearn.metrics import average_precision_score # площадь под precision recall кривой
average_precision_score (training_values, cross_validation_predictions) 
test_points
test_predictions = xgboost_model.predict_proba(test_points)[:, 1]
result = pd.DataFrame(columns=['index', 'Attrition'])
result['index'] = ids

result['Attrition'] = test_predictions
result.to_csv('result.csv', index=False)
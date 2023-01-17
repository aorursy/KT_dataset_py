# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import confusion_matrix, average_precision_score # метрики качества
%matplotlib inline
import matplotlib.pyplot as plt
training1 = "/kaggle/input/employee-resignation-mlpo/train_data.csv"
test1 = "/kaggle/input/employee-resignation-mlpo/test_data.csv"
training_data = pd.read_csv(training1)
training_data.describe().T
training_data.info()
training_data.hist(figsize=(20,14))
train_mean = training_data.mean()
train_mean
training_data.fillna(train_mean, inplace=True)
target_variable_name = 'Attrition'
training_values = training_data[target_variable_name]
training_points = training_data.drop(target_variable_name, axis=1)
training_points.shape
training_points.describe().T
training_points.hist(figsize=(12,8))
training_values.value_counts(normalize=True)
test_data = pd.read_csv(test1)
test_data.fillna(train_mean, inplace=True)
ids = test_data['index'] # записываем столбец id в отдельную переменную
test_points = test_data.drop('index', axis=1) # удаляем его из тестовой выборки 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
text_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
label_encoder = LabelEncoder()
for col in text_features:
    training_points[col] = label_encoder.fit_transform(training_points[col]) + 1
    test_points[col] = label_encoder.transform(test_points[col]) + 1
import xgboost as xgb
xgboost_model = xgb.XGBClassifier(n_estimators=500)
xgboost_model.fit(training_points, training_values)
test_predictions = xgboost_model.predict_proba(test_points)[:,1]
result = pd.DataFrame(columns=['index', 'Attrition'])
result['index'] = ids
result['Attrition'] = test_predictions
result.to_csv('resultend3.csv', index=False)
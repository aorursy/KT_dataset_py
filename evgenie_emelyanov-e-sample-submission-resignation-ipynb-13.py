import pandas as pd
from sklearn.metrics import confusion_matrix, average_precision_score # метрики качества
%matplotlib inline
import matplotlib.pyplot as plt
%%capture
!wget https://www.dropbox.com/s/z72mi1fh0cmadcv/train_data.csv
!wget https://www.dropbox.com/s/7mzi72914hpo74t/test_data.csv
training_data = pd.read_csv('train_data.csv')
training_data.describe().T
training_data.info()
train_mean = training_data.mean()
train_mean
training_data.fillna(train_mean, inplace=True)
target_variable_name = 'Attrition'
training_values = training_data[target_variable_name]
training_points = training_data.drop(target_variable_name, axis=1)
training_points.shape
test_data = pd.read_csv('test_data.csv')
test_data.fillna(train_mean, inplace=True)
ids = test_data['index'] # записываем столбец id в отдельную переменную
test_points = test_data.drop('index', axis=1) # удаляем его из тестовой выборки 
from sklearn.preprocessing import LabelEncoder
text_features = ['BusinessTravel', 'Department', 'EducationField', 
                 'Gender', 'JobRole', 'MaritalStatus']
label_encoder = LabelEncoder()
for col in text_features:
    training_points[col] = label_encoder.fit_transform(training_points[col]) + 1
    test_points[col] = label_encoder.transform(test_points[col]) + 1
import xgboost as xgb
xgboost_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.01)
xgboost_model.fit(training_points, training_values)
test_predictions = xgboost_model.predict_proba(test_points)[:,1]
result = pd.DataFrame(columns=['index', 'Attrition'])
result['index'] = ids
result['Attrition'] = test_predictions
result.to_csv('result.csv', index=False)
from google.colab import files
files.download('result.csv')

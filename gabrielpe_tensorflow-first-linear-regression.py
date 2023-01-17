import pandas as pd
data = pd.read_csv('../input/Salary_Data.csv') 
data.head()
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/Salary_Data.csv') 
X = np.array(data['YearsExperience'])
y = np.array(data['Salary'])

#reduce salaries to unit of thousands
y = y/1000
#Split 70% training, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Create estimator
feat_cols = [ tf.feature_column.numeric_column('X', shape=[1]) ]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

#input functions
train_input_func = tf.estimator.inputs.numpy_input_fn({'X': X_train}, y_train, shuffle=False)
test_input_func = tf.estimator.inputs.numpy_input_fn({'X': X_test}, y_test, shuffle=False)

#Train and test
estimator.train(input_fn=train_input_func)
train_metrics = estimator.evaluate(input_fn=train_input_func)
test_metrics = estimator.evaluate(input_fn=test_input_func)

#Predict salary for arbitrary years of experience
X_single_data = np.array([4.6])
pred_input_func = tf.estimator.inputs.numpy_input_fn({'X': X_single_data}, shuffle=False)
single_pred = estimator.predict(pred_input_func)

print('--Train metrics--')
print(train_metrics)
print(' ')
print('--Test metrics--')
print(test_metrics)

for result in single_pred:
    print (f'{result}')
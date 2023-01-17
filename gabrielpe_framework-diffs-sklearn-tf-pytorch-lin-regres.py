import pandas as pd
data = pd.read_csv('../input/Salary_Data.csv')
data.head()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#Read and split data set
data = pd.read_csv('../input/Salary_Data.csv') 
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

#Split 70% training, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
#Perform regression on dataset
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#Predict salary for arbitrary years of experience
X_single_data = [[4.6]]
y_single_pred = regressor.predict(X_single_data)

#Plot Result
splt = plt.subplot()
plt.title('Salary vs Years of experience')
plt.xlabel('Years')
plt.ylabel('Salary')
splt.plot(X_test, y_pred, color='blue', alpha=0.5, label='Test data(trend)')
splt.scatter(X_test, y_test, color='green', alpha=0.5, label='Test data')
splt.scatter(X_train, y_train, color='red', alpha=0.5, label='Train data')
splt.scatter(X_single_data, y_single_pred, color='black', label='Single prediction')
splt.legend(loc='upper left')

#Print performance
print(f'Train score: {regressor.score(X_train, y_train)}')
print(f'Test  score: {regressor.score(X_test, y_test)}')
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/Salary_Data.csv') 
X = np.array(data['YearsExperience'])
y = np.array(data['Salary'])
 
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/jiffs-house-price-prediction-dataset/jiffs_house_price_my_dataset_v1.csv')
df.head()
df.info()
df.describe().transpose()
X = df.drop('property_value', axis=1)
y = df['property_value']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
X_train.shape
X_test.shape
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn import metrics

print('MAE:', round(metrics.mean_absolute_error(y_test, predictions),2))
print('MSE:', round(metrics.mean_squared_error(y_test, predictions),2))
print('RMSE:', round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2))
index_position = 1
np_y_test = np.array(y_test)
actual = np_y_test[index_position]/1
pred = round(predictions[index_position],2)
diff = round((np_y_test[index_position]/1)-(predictions[index_position]),2)
perc = round(diff/actual*100,2)
print('Actual is: ' + str(actual))
print('Prediction is: ' + str(pred))
print('Difference is: ' + str(diff))
print('Error Percentage is: ' + str(perc)+'%')
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression #Linear Regression model
from sklearn.metrics import mean_squared_error #Function to calculate RMSE
from sklearn.model_selection import train_test_split
sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


raw_data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
raw_data.head()
raw_data.describe()
raw_data = raw_data.drop(['Serial No.'],axis = 1)
data = raw_data.copy()
data
target = data['Chance of Admit ']
inputs = data.drop(['Chance of Admit '],axis = 1)
inputs
x_train = inputs[0:399]
y_train = target[0:399]
x_test = inputs[400:399]
y_test = target[400:399]
reg = LinearRegression()
reg.fit(inputs,target)
true_value = target.loc[400:499]
predictions = pd.DataFrame({'Chance of Admit ':reg.predict(inputs.loc[400:499])})
predictions = predictions.round(2)
predictions
reg.coef_
plt.scatter(true_value,predictions,alpha = 0.8)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show()
reg_summary = pd.DataFrame(inputs.columns.values,columns = ['Classes'])
reg_summary['Weights']= reg.coef_
reg_summary.round(3)
print('This model has an accuracy rate of',(reg.score(inputs,target)*100).round(3),'%')
output = pd.DataFrame({'Chance of Admit ':reg.predict(inputs)})
output.to_csv('my_submission.csv')
mean_squared_error(true_value,predictions)
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
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import os
print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')
df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
df_test.head()
df_train.head()
correlation_matrix = df_train.corr()
sns.heatmap(correlation_matrix,annot = True)
plt.show()
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)
train_X = df_train.loc[:, 'F3': 'F17']
train_y = df_train.loc[:, 'O/P']
train_X
X_train,X_valid,y_train,y_valid = train_test_split(train_X,train_y,train_size = 0.9,test_size=0.1)
X_train
from sklearn.metrics import mean_absolute_error 
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
learning_rate_values = [0.05,0.1,0.25,0.5]           # different values of learning_rate to see which one performs best
mse_values = []                                   # corresponding mae values

for i in learning_rate_values:
    xgb = XGBRegressor(num_estimators = 1000, learning_rate = i) 
    xgb.fit(X_train,y_train)                         # fitting the model on the training data
    predictions = xgb.predict(X_valid)               # predicting the price on the validation data
    mse = mean_squared_error(predictions,y_valid)                                                     
    mse_values.append(np.sqrt(mse))
    print(np.sqrt(mse))
# plotting a graph of learning_rate v/s mae_values
plt.plot(learning_rate_values,mse_values)
plt.xlabel('Learning rate')
plt.ylabel('Mean-absolute error')
plt.show()
from sklearn.ensemble import RandomForestRegressor
max_depth_values = [10,100,150,1000,1500]
mse_values = []
for i in max_depth_values:
    rf_model = RandomForestRegressor(max_depth=i,max_leaf_nodes =100,n_estimators=100)
    rf_model.fit(X_train,y_train)                                                     # training the model on the dataset
    predictions  = rf_model.predict(X_valid)                                          # making predictions 
    mse = mean_squared_error(predictions,y_valid)                                                     
    mse_values.append(np.sqrt(mse))
    print(np.sqrt(mse))
# plotting the grapgh of max_depth v/s corresponding mae values
plt.plot(max_depth_values,mse_values)   
plt.xlabel('Max_depth')
plt.ylabel('Mean-absolute error')
plt.show()
from sklearn import linear_model
algorithms = ['xgboost','random forest','linear regression']
mae_values = []
mse_values = []

# xgboost model
xgb = XGBRegressor(num_estimators = 1000, learning_rate = 0.25) 
xgb.fit(X_train,y_train)                         # fitting the model on the training data
predictions = xgb.predict(X_valid)               # predicting the price on the validation data
mae = mean_absolute_error(predictions,y_valid)   # finding mae to check accuracy
mse = mean_squared_error(predictions,y_valid)    # finding mse
mse_values.append(np.sqrt(mse))
mae_values.append(mae)


# random forest model
rf_model = RandomForestRegressor(max_depth=150,max_leaf_nodes =100,n_estimators=100)
rf_model.fit(X_train,y_train)
predictions  = rf_model.predict(X_valid)
mae = mean_absolute_error(predictions,y_valid)
mse = mean_squared_error(predictions,y_valid)     
mse_values.append(np.sqrt(mse))
mae_values.append(mae)

# linear regression model

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
predictions  = rf_model.predict(X_valid)
mae = mean_absolute_error(predictions,y_valid)
mse = mean_squared_error(predictions,y_valid)     
mse_values.append(np.sqrt(mse))
mae_values.append(mae)


# plotting the graph between algorithm v/s mse

plt.bar(algorithms,mse_values)
plt.xlabel('Different algorithms')
plt.ylabel('Mean-squared error')
plt.show()
result=pd.DataFrame()
result['Id'] = df_test['index']
df_test = df_test.loc[:, 'F3':'F17']
pred = xgb.predict(df_test)
print(pred)
result['PredictedValue'] = pd.DataFrame(pred)
result.head()
result.to_csv('output.csv', index=False)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize']=10,8
matplotlib.style.use('ggplot')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
data = pd.DataFrame({
    'x':x,
    'y':y
})
data.head()
data.corr()
data.plot(kind="scatter",
x='x',
y='y',
color = "purple",
figsize=(10,8)
)
plt.ylabel("Y - Target Variable", size=15)
plt.xlabel('X - Independent Variable',size=15)
plt.suptitle('Fig. 1 showing the correlation between X and Y', size=15, color='black')

# use k-1 subsets of data to train our model and leave  the last fold as test data
from sklearn.model_selection import KFold
x_data_cv = x
y_data_cv = y
kf = KFold(n_splits=5,shuffle=True) #SPLIT DATA INTO 5 FOLDS AND SHUFFLE BEFORE SPLIT
for train_index, test_index in kf.split(x_data_cv):
    print("TRAIN:\n", train_index, "\nTEST:", test_index)
    X_train_cv, X_test_cv = x_data_cv[train_index], x_data_cv[test_index]
    y_train_cv, y_test_cv = y_data_cv[train_index], y_data_cv[test_index]

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
X_train_df_cv =pd.DataFrame(X_train_cv)
model_cv=reg.fit(X_train_df_cv,y_train_cv) #fit model to training dataset
X_test_df_cv = pd.DataFrame(X_test_cv)
predictions_cv=reg.predict(X_test_df_cv)
predictions_cv #testing dataset
data.plot(kind="scatter",
x='x',
y='y',
color = "blue",
figsize=(10,8)
)
plt.ylabel("y", size=15)
plt.xlabel('x',size=15)
plt.suptitle('Fig. 2 showing linear regression', size=15, color='black')
plt.plot(data.x,reg.predict(data[['x']]), color='purple')
#CHECKING PREDICTIONS
data2 = X_test_df_cv.copy()
data2['Actual_y_value']=y_test_cv
data2['Predicted_y_value']=predictions_cv
data2
plt.scatter(data.x, y_data_cv - reg.predict(data[['x']]))
plt.hlines(y=0,xmin=-1.5,xmax=11,color='purple',linewidth=3)
plt.suptitle('Fig. 3 showing Residual Plot ',size = 15 )
plt.ylabel('Residuals', size=15)
plt.xlabel('X', size=15)
print('Score: ', reg.score(X_test_df_cv,y_test_cv))
#Intercept
intercept = model_cv.intercept_
#Regression Coefficient
slope = reg.coef_
#R**2
gof = reg.score(X_test_df_cv,y_test_cv)
line = pd.DataFrame({'y-intercept':intercept,'Regression Coefficient':slope,'Coefficient of Determination':gof})
line

def regFormula(x,y):
    x_df = pd.DataFrame(x)
    y_df = pd.DataFrame(y)
    y_pred = (slope * x_df) + intercept #formula
    results = x_df.copy()
    results["Actual_y_value"] =y_df
    results['Predicted_y_value']=y_pred
    #print(results.head(10)) #only show first 10 records
    return results
x2 = 5 * rng.rand(50) #create new x-value
data2 = pd.DataFrame(x2, columns=['x2'])
df_xy_2 = pd.concat([data,data2], axis =1 )
df_xy_2.head()

#predictions for 2nd x-value
x2_pred = regFormula(df_xy_2.x2,df_xy_2.y)
x2_pred.head()
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
#MAE
print("Mean Absolue Error:%.2f"%mean_absolute_error(x2_pred['Actual_y_value'],x2_pred['Predicted_y_value']))
#RMSE                                                                  
print("Root Mean Squared Error:%.2f"%sqrt(mean_squared_error(x2_pred['Actual_y_value'],x2_pred['Predicted_y_value'])))
#MSE
print("Root Mean Squared Error:%.2f"%mean_squared_error(x2_pred['Actual_y_value'],x2_pred['Predicted_y_value']))
                                                                                                
                                                                                                
                                                                                                

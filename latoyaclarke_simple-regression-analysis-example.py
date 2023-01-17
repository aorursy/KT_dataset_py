#necessary libraries
import numpy as np
import pandas as pd
from pandas import  DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize']=10,8 
rng = np.random.RandomState(1)
x = 10 * rng.rand(100)
y = 2 * x - 5 + rng.randn(100)
df_xy = pd.DataFrame({'X1':x, 'Y':y})
#df_xy = pd.concat([df_x,df_y], axis = 1)

x_data=df_xy['X1']
y_data=df_xy['Y']
df_xy.head()
x_data.corr(y_data)
df_xy.plot(kind="scatter",
    x='X1',
    y='Y',
    color = "b",
    figsize=(10,8)
)
plt.ylabel("Y - Target Variable", size=15)
plt.xlabel('X1 - Independent Variable',size=15)
plt.suptitle('Fig. 2 Showing: Correlation between X1 and Y', size=20, color='black')
from sklearn.model_selection import KFold 

kf = KFold(n_splits=10,shuffle=True) #SPLIT DATA INTO 10 FOLDS AND SHUFFLE BEFORE SPLIT

for train_index, test_index in kf.split(x_data):
   # print("TRAIN:\n", train_index, "\nTEST:", test_index)
    X_train, X_test = x_data[train_index], x_data[test_index]
    
    y_train, y_test = y_data[train_index], y_data[test_index]
   
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

X_train_df=pd.DataFrame(X_train)
reg.fit(X_train_df,y_train) #fit model to training dataset

reg.coef_ 
X_test_df = pd.DataFrame(X_test)
prediction = reg.predict(X_test_df)
prediction
df_xy.plot(kind="scatter",
    x='X1',
    y='Y',
    color = "b",
    figsize=(10,8)
)
plt.ylabel("Y", size=15)
plt.xlabel('X',size=15)
plt.suptitle('Fig. 2 Showing: Impact of X1 on Y', size=20, color='black')
plt.plot(df_xy.X1,reg.predict(df_xy[['X1']]), color='green')
# display predictions and residuals
data2 = X_test_df.copy()
data2['Actual_y_value']=y_test
data2['Predicted_y_value']=prediction
data2['Residuals'] = y_test - prediction


data2.head()
# plotting residuals
plt.scatter(df_xy.X1, y_data- reg.predict(df_xy[['X1']]))
plt.hlines(y=0,xmin=-1.5,xmax=11,color='red',linewidth=3)
plt.suptitle('Fig. 3 Showing: Residual Plot ',size = 20 )
plt.ylabel('Residuals', size=15)
plt.xlabel('X1', size=15)
#CHECKING ACCURACY SCORE R^2
reg.score(X_test_df,y_test)
#Intercept
intercept = reg.intercept_
#Regression Coefficient
slope = reg.coef_
#R**2
gof = reg.score(X_test_df,y_test) # goodness of fit

line = pd.DataFrame({'y-intercept':intercept, 'Regression Coefficient':slope, 'Coefficient of Determination':gof})
line
#Formula
#y=mx+c
def regFormula(x):
    x_df = DataFrame(x)
    y_pred = (slope * x_df) + intercept #formula
    results = x_df.copy()
    results['X2-Predicted Values']=y_pred
    #print(results.head(10)) #only show first 10 records
    return results
from sklearn.metrics import mean_squared_error,mean_absolute_error
from math import sqrt
#Mean Absolute Error
print("Mean Absolue Error:%.2f"%mean_absolute_error(y_test,prediction))
#Root Mean Squared Error
print("Root Mean Squared Error:%.2f"%sqrt(mean_squared_error(y_test,prediction)))
#Mean Squared Error
print("Mean Squared Error:%.2f"%mean_squared_error(y_test,prediction))
x2 = 5 * rng.rand(100) #generate new x-value
df_x2 = DataFrame(x2, columns=['X2'])
df_xy_2 = pd.concat([df_xy,df_x2], axis =1 ) # concatenate new x-values wiith previous dataframe
df_xy_2.head()

#predictions for 2nd x-value
x2_pred = regFormula(df_xy_2.X2)
x2_pred.head()
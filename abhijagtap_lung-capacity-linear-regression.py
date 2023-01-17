#import all the reuqired libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

lc=pd.read_csv(r"//kaggle/input/lung-capacity-smoker-and-non-smoker/Lung-Capacity-Smoker.csv")
lc.head()
lc.shape
#hot encoding for categorical variables
lc.Gender.replace({"male":1,"female":0},inplace=True)
lc.Smoke.replace({"yes":1,"no":0},inplace=True)
lc.Caesarean.replace({"yes":1,"no":0},inplace=True)
#check for null values
lc.isnull().sum()
lc_x=lc.iloc[:,1:6]
lc_y=lc.iloc[:,0]
#Sampling of data into train & test
import sklearn

from sklearn.model_selection import train_test_split
lc_x_train, lc_x_test, lc_y_train, lc_y_test=train_test_split(lc_x,lc_y,test_size=0.2, random_state=101)
#linear modelling & Prediction
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(lc_x_train,lc_y_train) #training the algorithm
pred_val=reg.predict(lc_x_test)
pred_val
#Calulate the best values for intercept & slope
print(reg.coef_)
#slope
print(reg.intercept_)
#R-Square
reg.score(lc_x_train,lc_y_train)
## Convert test data and predicted data in Series for concatenation
X= []
for i in lc_y_test:
    X.append(i)
X = pd.Series(X)
    
pred_val= pd.Series(pred_val)
pred_val
## compare actual and predicted values
final = pd.concat({"Actual" : X,"Predicted": pred_val}, axis=1,join='outer')
final
print('Mean Absolute Error:', metrics.mean_absolute_error(lc_y_test, pred_val))  
print('Mean Squared Error:', metrics.mean_squared_error(lc_y_test, pred_val))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(lc_y_test,pred_val)))
#check the difference between actual & predicated values
df = pd.DataFrame({'Actual': X, 'Predicted': pred_val})
df1 = df.head(25)
#Plot the actual & predicated values
import matplotlib.pyplot as plt
%matplotlib inline
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

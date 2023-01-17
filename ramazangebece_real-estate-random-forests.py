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
df=pd.read_csv('../input/real-estate-price-prediction/Real estate.csv')
df.head()
df=df.drop(['No','X1 transaction date'],axis=1)
df.head()
#change to name of columns:
df.rename({'X2 house age':'house age','X3 distance to the nearest MRT station':'station',
          'X4 number of convenience stores':'stores','X5 latitude':'latidude','X6 longitude':'longitude',
          'Y house price of unit area':'price'},axis='columns',inplace=True)
df.head(10)
#this dataser 414 observation units and 6 columns
df.shape
df.info()
#have the dataset null values any observation units?
df.isnull().sum().sum()
import seaborn as sns
import matplotlib.pyplot as plt
#corelation of columns:
plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),annot=True,fmt=".3f",linewidths = .5);
#descriptive statistics of columns
plt.figure(figsize=(17,7))
sns.heatmap(df.describe().T,annot=True,fmt='.2f',linewidths = .5)
plt.show()
df.head()
#add independent variables to x 
#add dependent variable to y
x=df.drop(['price'],axis=1)   #ı erased dependent variable by .drop funciton,on to exist dependent variable in x valuable
y=df['price']    #this is dependent variable in y valuable
#independent variables
x.head()
#dependent variable
y[0:5]
#now ı will train-test split by function of train_test_split
from sklearn.model_selection import train_test_split
#ready:%90 for train,%10 for test 
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                              test_size=0.10,
                                              random_state=42)
print('x_train_shape',x_train.shape)
print('x_test_shape',x_test.shape)
print('y_train_shape',y_train.shape)
print('y_test_shape',y_test.shape)
#now,ı import RandomForestRegressor function:
from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor(random_state=42)  #model object
rf_model.fit(x_train,y_train)   #model was fitted
#predict:
rf_model.predict(x_test)
y_pred=rf_model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
print(np.sqrt(mean_squared_error(y_test,y_pred)))
print(r2_score(y_test,y_pred))
rf_params={'max_depth':list(range(1,10)),
           'max_features':[3,5,10,15],
           'n_estimators':[100,200,500,1000,2000]}
rf_model=RandomForestRegressor(random_state=42)
from sklearn.model_selection import GridSearchCV
rf_cv_model=GridSearchCV(rf_model,
                         rf_params,
                         cv=10,
                         n_jobs=-1).fit(x_train,y_train)
#optimum parameter:
rf_cv_model.best_params_
#now,ı will buil final model,to use best_params:
rf_tuned=RandomForestRegressor(max_depth=rf_cv_model.best_params_['max_depth'],
                              max_features=rf_cv_model.best_params_['max_features'],
                              n_estimators=rf_cv_model.best_params_['n_estimators']).fit(x_train,y_train)
#predict:
rf_tuned.predict(x_test)
y_pred=rf_tuned.predict(x_test)
#test error and r*2 score:
print('test error:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('r*2 score:',r2_score(y_test,y_pred))
import matplotlib.pyplot as plt
#let's see importance of variables:
#significance according to dependent variable:
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                          index = x_train.columns)
Importance
Importance.sort_values(by = "Importance",
axis = 0,
ascending = True).plot(kind ="barh", color = "r")
plt.xlabel("significance of independent variables")
plt.show()

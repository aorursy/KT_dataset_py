# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_housing = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
df_housing.dtypes
len (df_housing)
df_housing['ocean_proximity'].value_counts()
df_housing['ocean_proximity'] = df_housing['ocean_proximity'].convert_dtypes()
df_housing.dtypes
for col in df_housing.columns:
    print(col)
    print(df_housing[col].isnull().value_counts())
    print('#####################')
df_housing.dropna(inplace = True)
for col in df_housing.columns:
    print(col)
    print(df_housing[col].isnull().value_counts())
    print('#####################')

df_housing = pd.concat([df_housing,pd.get_dummies(df_housing['ocean_proximity'])],axis =1)
df_housing.drop('ocean_proximity',axis =1)
import seaborn as sns
import matplotlib.pyplot as plt
figure = plt.figure(figsize=(12,10))
sns.heatmap(df_housing.corr(),annot =True)
df_housing.corr().loc["median_house_value"] >=0.1
df_housing.corr().loc["median_house_value"] <=-0.1
df_data=df_housing[['housing_median_age','total_rooms','median_income','median_house_value','<1H OCEAN','NEAR BAY','NEAR OCEAN','INLAND','latitude']]
for col in ['housing_median_age','total_rooms','median_income','median_house_value','latitude'] :
    df_data[col] =  df_data[col]/ df_data[col].max() 
  
df_data.head()
train, validate, test = np.split(df_data.sample(frac=1), [int(.6*len(df_data)), int(.8*len(df_data))])
X_Train = train[['housing_median_age','total_rooms','median_income','<1H OCEAN','NEAR BAY','NEAR OCEAN','INLAND','latitude']]
Y_Train = train[['median_house_value']]
X_Val =validate[['housing_median_age','total_rooms','median_income','<1H OCEAN','NEAR BAY','NEAR OCEAN','INLAND','latitude']]
Y_Val = validate[['median_house_value']]
X_Test =test[['housing_median_age','total_rooms','median_income','<1H OCEAN','NEAR BAY','NEAR OCEAN','INLAND','latitude']]
Y_Test = test[['median_house_value']]
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt 

reg_model = LinearRegression().fit(X_Train,Y_Train)
Y_hat_Val = reg_model.predict(X_Val)
print("Mean Squared Error on Validation Set is : ",mean_squared_error(Y_Val,Y_hat_Val))
print("R2 score on Validation Set is : ",r2_score(Y_Val,Y_hat_Val))
Y_hat_Test = reg_model.predict(X_Test)
print("Mean Squared Error on Test Set is : ",mean_squared_error(Y_hat_Test,Y_Test))
print("R2 score on Validation Set is : ",r2_score(Y_hat_Test,Y_Test))
Y_hat_Train = reg_model.predict(X_Train)
print("Mean Squared Error on Train Set is : ",mean_squared_error(Y_hat_Train,Y_Train))
print("R2 score on Validation Train is : ",r2_score(Y_hat_Train,Y_Train))
X_Train = train[['housing_median_age','total_rooms','median_income','<1H OCEAN','NEAR BAY','NEAR OCEAN','INLAND','latitude']]
Y_Train = train[['median_house_value']]
splited_train_X = np.split(X_Train,13)
splited_train_Y = np.split(Y_Train,13)
error_list_val =[]
error_list_train =[]
train_len_list =[]
train_set_len= 0
for X,Y in zip(splited_train_X,splited_train_Y) :
    train_set_len = train_set_len + len(X)
    reg_model = LinearRegression().fit(X,Y)
    
    y_hat_val = reg_model.predict(X_Val)
    y_hat_train = reg_model.predict(X)
    
    error_list_val.append(mean_squared_error(y_hat_val,Y_Val))
    error_list_train.append(mean_squared_error(y_hat_train,Y))
    
    train_len_list.append(train_set_len)
df_error_log_CV = pd.DataFrame({'Number of Training Sample': train_len_list,'Cross Validation Error': error_list_val})
df_error_log_Train = pd.DataFrame({'Number of Training Sample': train_len_list,'Training Error': error_list_train})
fig = plt.figure(figsize = (8,5))
ax = fig.add_subplot(111)
df_error_log_Train.plot(x='Number of Training Sample',y='Training Error',ax=ax)
df_error_log_CV.plot(x='Number of Training Sample',y='Cross Validation Error',ax=ax)   
reg_model = LinearRegression().fit(X,Y)
y_hat_test = reg_model.predict(X_Test)
print("MSE values on test set :",mean_squared_error(y_hat_test,Y_Test))
print("r2 score on test set :",r2_score(y_hat_test,Y_Test))
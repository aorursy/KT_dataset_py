# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



 # linear algebra

 # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error
diabetes = datasets.load_diabetes() 

diabetes.keys()  # to find the content of data
diabetes["feature_names"]
df = pd.DataFrame(diabetes['data'],columns = diabetes['feature_names']) #putting our data in a Dataframe
df.head()     #Shows top 5 data entry
df.describe()   # Decribe about the data 
df.isnull().sum() 

#checking for any sort of null value in our data and return the sum of no. null values 

#Remove .sum() and it will so False if null value is not present for every single record 
df.count()   #tells us the number of entries in each column  
x = df          

y = diabetes['target']
from sklearn.model_selection import train_test_split #to split our data into training and testing set



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101) #splitting our data
from sklearn import  linear_model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)  # Training data is used always



# Prediction of testset result of  the Prepared Model

y_pre = model.predict(x_test)   # puts the test feature value to get the label value which are predicted by the model



from sklearn.model_selection import cross_val_score    #importing 

scores = cross_val_score(model,x,y,scoring="neg_mean_squared_error" , cv=10)  

rmse_scores=np.sqrt(-scores).mean()    #calculating  root mean sq. of the resulted scores of array 

rmse_scores
from sklearn.metrics import r2_score

r2_score(y_test, y_pre)
mse=mean_squared_error(y_test, y_pre)

rmse=np.sqrt(mse)

rmse
print("Weights:",model.coef_)

print("\nIntercept",model.intercept_)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model,x,y,scoring="neg_mean_squared_error" , cv=10)

rmse_scores=np.sqrt(-scores).mean()

rmse_scores
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
Ins_Data = pd.read_csv("../input/insurance.csv")
Ins_Data
Ins_Data.tail()
Ins_Data.head()
#from sklearn.model_selection import train_test_split

#test_x,train_x,test_y,train_y = train_test_split(x,y,est_size = 0)
Ins_Data.describe().T
Ins_Data.describe
#graphs = sns.pairplot(data)

#graphs.set()
Ins_Data.info()
Ins_Data["sex"].value_counts()
Ins_Data["sex"]= Ins_Data.sex.replace({'male':1,'female':2})
Ins_Data["sex"]
Ins_Data["smoker"].value_counts()
Ins_Data["smoker"]= Ins_Data.smoker.replace({'yes':1,'no':0})
Ins_Data["smoker"]
Ins_Data["region"].value_counts()
Ins_Data["region"]= Ins_Data.region.replace({'southeast':1,'southwest':2,'northwest':3,'northeast':4})
Ins_Data["region"]
Ins_Data.head()
Ins_Data.tail()
sns.pairplot(data=Ins_Data)
Ins_Data.corr()
#Data Analysis

#Age vs c harges

#The more age will have more charges(roughly estimated)

plt.figure(figsize = (12, 8))

sns.barplot(x = 'age', y = 'expenses', data = Ins_Data)



plt.title("Age vs expenses")


# sex vs expenses

# males have slightly greater insurance than females in generally(Assumption)



plt.figure(figsize = (6, 6))

sns.barplot(x = 'sex', y = 'expenses', data = Ins_Data)



plt.title('sex vs expenses')
# children vs Expenses

# no. of childrens of a person has a very interesting dependency on insurance costs



plt.figure(figsize = (12, 8))

sns.barplot(x = 'children', y = 'expenses', data = Ins_Data)



plt.title('children vs expenses')
# region vs expenses

# From the graph we can see that the region actually does not play any role in determining the insurance 



plt.figure(figsize = (12, 8))

sns.barplot(x = 'region', y = 'expenses', data = Ins_Data)



plt.title('region vs expenses')
# smoker vs expenses

# from the graph below, it is visible that smokers have more expenses than the non smokers



plt.figure(figsize = (6, 6))

sns.barplot(x = 'smoker', y = 'expenses', data = Ins_Data)



plt.title('smoker vs expenses')
Ins_Data.corr()
#splitting dependednt and independednt variable

y = Ins_Data["expenses"]

x = Ins_Data.drop(columns=["expenses","sex","region"])



print(y.shape)

print(x.shape)


# splitting the dataset into training and testing sets



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
# standard scaling



from sklearn.preprocessing import StandardScaler



# creating a standard scaler

sc = StandardScaler()



# feeding independents sets into the standard scaler

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score





# creating the model

model = LinearRegression()



# feeding the training data to the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2, axis = None)

print("MSE :", mse)



# Calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



# Calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2 score :", r2)
y_test
y_pred
# Print various metrics



from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



print("Predicting the train data")

train_predict = model.predict(x_train)

print("Predicting the test data")

test_predict = model.predict(x_test)

print("1.MAE")

print("Train : ",mean_absolute_error(y_train,train_predict))

print("Test  : ",mean_absolute_error(y_test,test_predict))

print("*****************************************")

print("2.MSE")

print("Train : ",mean_squared_error(y_train,train_predict))

print("Test  : ",mean_squared_error(y_test,test_predict))

print("*****************************************")

import numpy as np

print("3.RMSE")

print("Train : ",np.sqrt(mean_squared_error(y_train,train_predict)))

print("Test  : ",np.sqrt(mean_squared_error(y_test,test_predict)))

print("*****************************************")

print("4.R^2")

print("Train : ",r2_score(y_train,train_predict))

print("Test  : ",r2_score(y_test,test_predict))

print("*****************************************")

print("5.MAPE")

print("Train : ",np.mean(np.abs((y_train - train_predict) / y_train)) * 100)

print("Test  : ",np.mean(np.abs((y_test - test_predict) / y_test)) * 100)
#Plot actual vs predicted value

plt.figure(figsize=(10,7))

plt.title("Actual vs. predicted expenses",fontsize=25)

plt.xlabel("Actual expenses",fontsize=18)

plt.ylabel("Predicted expenses", fontsize=18)

plt.scatter(x=y_test,y=test_predict)
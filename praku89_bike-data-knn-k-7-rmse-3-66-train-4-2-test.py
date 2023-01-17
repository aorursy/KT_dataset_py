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
bike_df=pd.read_csv("../input/bike_share.csv")
bike_df.shape
bike_df.isna().sum()
bike_df[bike_df.duplicated()]
bike_df_unique=bike_df.drop_duplicates()

bike_df_unique.shape
bike_df_unique.info()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(20, 10))

sns.boxplot(data=bike_df_unique,orient="h")
bike_df_unique.describe().transpose()
lower_bnd = lambda x: x.quantile(0.25) - 1.5 * ( x.quantile(0.75) - x.quantile(0.25) )

upper_bnd = lambda x: x.quantile(0.75) + 1.5 * ( x.quantile(0.75) - x.quantile(0.25) )
bike_df_unique.shape
for i in bike_df_unique.columns:

    bike_df_clean = bike_df_unique[(bike_df_unique[i] >= lower_bnd(bike_df_unique[i])) & (bike_df_unique[i] <= upper_bnd(bike_df_unique[i])) ] 

bike_df_clean.shape
plt.figure(figsize=(20, 10))

sns.boxplot(data=bike_df_clean,orient="h")
bike_df_clean.corr()
#sns.heatmap(bike_df_clean,annot=True)
sns.pairplot(data=bike_df_clean)
num_disc_list=['season', 'holiday','workingday','weather']

num_cont_list=bike_df_clean.columns.drop(num_disc_list)

sns.pairplot(data=bike_df_clean,vars = num_cont_list,hue=num_disc_list[0])
#for i in num_disc_list:

 #   sns.pairplot(data=bike_df_clean,vars = num_cont_list,hue=i)
# Importing necessary package for creating model

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
#using one hot encoding

X=bike_df_clean.drop(columns='count')

y=bike_df_clean[['count']]
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=1234)
model = LinearRegression()



model.fit(train_X,train_y)
# Print Model intercept and co-efficent

print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)

cdf = pd.DataFrame(data=model.coef_.T, index=X.columns, columns=["Coefficients"])

cdf
# Print various metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



print("Predicting the train data")

train_predict = model.predict(train_X)

print("Predicting the test data")

test_predict = model.predict(test_X)

print("MAE")

print("Train : ",mean_absolute_error(train_y,train_predict))

print("Test  : ",mean_absolute_error(test_y,test_predict))

print("====================================")

print("MSE")

print("Train : ",mean_squared_error(train_y,train_predict))

print("Test  : ",mean_squared_error(test_y,test_predict))

print("====================================")

import numpy as np

print("RMSE")

print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))

print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))

print("====================================")

print("R^2")

print("Train : ",r2_score(train_y,train_predict))

print("Test  : ",r2_score(test_y,test_predict))

print("MAPE")

print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)

print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)
#Plot actual vs predicted value

plt.figure(figsize=(10,7))

plt.title("Actual vs. predicted count",fontsize=25)

plt.xlabel("Actual count",fontsize=18)

plt.ylabel("Predicted count", fontsize=18)

plt.scatter(x=test_y,y=test_predict)

#using one hot encoding

X=bike_df_unique.drop(columns='count')

y=bike_df_unique[['count']]
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=1234)
model = LinearRegression()



model.fit(train_X,train_y)
# Print Model intercept and co-efficent

print("Model intercept",model.intercept_,"Model co-efficent",model.coef_)

cdf = pd.DataFrame(data=model.coef_.T, index=X.columns, columns=["Coefficients"])

cdf
# Print various metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score



print("Predicting the train data")

train_predict = model.predict(train_X)

print("Predicting the test data")

test_predict = model.predict(test_X)

print("MAE")

print("Train : ",mean_absolute_error(train_y,train_predict))

print("Test  : ",mean_absolute_error(test_y,test_predict))

print("====================================")

print("MSE")

print("Train : ",mean_squared_error(train_y,train_predict))

print("Test  : ",mean_squared_error(test_y,test_predict))

print("====================================")

import numpy as np

print("RMSE")

print("Train : ",np.sqrt(mean_squared_error(train_y,train_predict)))

print("Test  : ",np.sqrt(mean_squared_error(test_y,test_predict)))

print("====================================")

print("R^2")

print("Train : ",r2_score(train_y,train_predict))

print("Test  : ",r2_score(test_y,test_predict))

print("MAPE")

print("Train : ",np.mean(np.abs((train_y - train_predict) / train_y)) * 100)

print("Test  : ",np.mean(np.abs((test_y - test_predict) / test_y)) * 100)
#Plot actual vs predicted value

plt.figure(figsize=(10,7))

plt.title("Actual vs. predicted count",fontsize=25)

plt.xlabel("Actual count",fontsize=18)

plt.ylabel("Predicted count", fontsize=18)

plt.scatter(x=test_y,y=test_predict)

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score

from math import sqrt
X_train, X_test, y_train, y_test = train_test_split(

X, y, test_size = 0.3, random_state = 100)

y_train=np.ravel(y_train)

y_test=np.ravel(y_test)
rmse_train_dict={}

rmse_test_dict={}

df_len=round(sqrt(len(bike_df_unique)))

#Train Model and Predict  

for k in range(3,df_len):

    neigh = KNeighborsRegressor(n_neighbors = k).fit(X_train,y_train)

    yhat_train = neigh.predict(X_train)

    yhat = neigh.predict(X_test)

    test_rmse=sqrt(mean_squared_error(y_test,yhat))

    train_rmse=sqrt(mean_squared_error(y_train,yhat_train))

    rmse_train_dict.update(({k:train_rmse}))

    rmse_test_dict.update(({k:test_rmse}))

    print("RMSE for train : ",train_rmse," test : ",test_rmse," difference between train and test :",abs(train_rmse-test_rmse)," with k =",k)
elbow_curve_train = pd.Series(rmse_train_dict,index=rmse_train_dict.keys())

elbow_curve_test = pd.Series(rmse_test_dict,index=rmse_test_dict.keys())

elbow_curve_test.head(10)
elbow_curve_train.head(10)
ax=elbow_curve_train.plot(title="RMSE of train VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("RMSE of train")
ax=elbow_curve_test.plot(title="RMSE of test VS Value of K ")

ax.set_xlabel("K")

ax.set_ylabel("RMSE of test")
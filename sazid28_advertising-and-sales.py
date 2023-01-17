import pandas as pd
data = pd.read_csv("../input/Advertising.csv",index_col = 0)
data.head(5)
import seaborn as sns
%matplotlib inline

sns.pairplot(data,x_vars = ["TV","radio","newspaper"],y_vars=["sales"],size = 7,aspect = 0.7,kind = "reg")
# for all 3 features

feature_cols_0 = ["TV","radio","newspaper"]
X = data[feature_cols_0]
X = data[["TV","radio","newspaper"]]         

feature_cols1_0 =["sales"]
Y = data[feature_cols1_0]
Y =data[["sales"]]
# test and train split
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X ,Y)

# create linear model

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
# find out intercept and coeff for 3 features
print (model.intercept_)
print(model.coef_)
#print for 3 features
pred_0 = model.predict(X_test)
print(pred_0)
from sklearn import metrics
import numpy as np

#calculate RMSE for 3 features
RMSE = np.sqrt(metrics.mean_squared_error(pred_0,Y_test))
print(RMSE)
#for TV and sales

feature_cols_0 = ["TV"]
X = data[feature_cols_0]
X = data [["TV"]]

feature_cols1_0 = ["sales"]
Y = data[feature_cols1_0]
Y = data[["sales"]]
#Y =data["sales"]
#Y = data.sales
#split train and test for tv and sales
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
model.fit(X_train,Y_train)
print(model.intercept_)
print(model.coef_)
pred_0 = model.predict(X_test)
print(pred_0)
RMSE = np.sqrt(metrics.mean_squared_error(pred_0,Y_test))
print(RMSE)
#for radio and sales

feature_cols_0 = ["radio"]
X = data[feature_cols_0]
X = data [["radio"]]

feature_cols1_0 = ["sales"]
Y = data[feature_cols1_0]
Y = data[["sales"]]
#split train and test for tv and sales
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
model.fit(X_train,Y_train)
print(model.intercept_)
print(model.coef_)
pred_0 = model.predict(X_test)
RMSE = np.sqrt(metrics.mean_squared_error(pred_0,Y_test))
print(RMSE)
#for newspaper and sales

feature_cols_0 = ["newspaper"]
X = data[feature_cols_0]
X = data [["newspaper"]]

feature_cols1_0 = ["sales"]
Y = data[feature_cols1_0]
Y = data[["sales"]]
#split train and test for tv and sales
X_train,X_test,Y_train,Y_test = train_test_split(X,Y)
model.fit(X_train,Y_train)
print(model.intercept_)
print(model.coef_)
pred_0 = model.predict(X_test)
RMSE = np.sqrt(metrics.mean_squared_error(pred_0,Y_test))
print(RMSE)
list(zip(feature_cols_0,model.coef_))

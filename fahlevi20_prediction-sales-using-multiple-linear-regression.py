import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv("../input/real-estate-price-prediction/Real estate.csv")
data.head()
data.info()
data.drop("No",axis=1,inplace=True)
data.head()
data.isnull().sum()
sns.distplot(data)
data.info()
data.info()
data=data.rename({"X1 transaction date":"transaction_date",
                       "X2 house age":"house_age",
                       "X3 distance to the nearest MRT station":"distanceTotheNearestMRT_station",
                       "X4 number of convenience stores":"Number_convenienceStores",
                       "X5 latitude":"latitude",
                       "X6 longitude":"longitude",
                       "Y house price of unit area":"house_price" },axis="columns")
data.info()
sns.pairplot(data,x_vars=["transaction_date","house_age","latitude","longitude","distanceTotheNearestMRT_station"],y_vars="house_price",kind="reg")
data.info()
sns.pairplot(data,kind="reg")
sns.heatmap(data.corr(),annot=True)
x= data[["transaction_date","house_age","distanceTotheNearestMRT_station","Number_convenienceStores","latitude","longitude"]]
y = data.house_price
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
X_train,X_test,Y_train,Y_test = train_test_split(x,y,random_state=1)
print(Y_test.shape)
model=LinearRegression().fit(X_train,Y_train)
model_pred = model.predict(X_test)

print("intercept",model.intercept_)
print("coef_:",model.coef_)
print("r2_score:",r2_score(Y_test,model_pred))
print("RMSE",np.sqrt(mean_squared_error(Y_test,model_pred)))
from yellowbrick.regressor import PredictionError,ResidualsPlot

visualizer = PredictionError(model)
visualizer.fit(X_train,Y_train)
visualizer.score(X_test,Y_test)

visualizer.poof()
visualizer_res=ResidualsPlot(model)
visualizer_res.fit(X_train,Y_train)
visualizer_res.score(X_test,Y_test)
visualizer.poof()
data["interaction"] = data["transaction_date"]*data["house_age"]*data["distanceTotheNearestMRT_station"]*data["Number_convenienceStores"]*data["latitude"]*data["longitude"]
x=data[["transaction_date","house_age","distanceTotheNearestMRT_station","Number_convenienceStores","latitude","longitude","interaction"]]
y=data.house_price

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 1)

lm3 = LinearRegression()
lm3.fit(X_train, y_train)
lm3_preds = lm3.predict(X_test)

print("RMSE :", np.sqrt(mean_squared_error(y_test, lm3_preds)))
print("R^2: ", r2_score(y_test, lm3_preds))
visualizer = PredictionError(lm3)
visualizer.fit(X_train,Y_train)
visualizer.score(X_test,Y_test)

visualizer.poof()


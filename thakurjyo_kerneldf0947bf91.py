import pandas as pd
df=pd.read_csv("../input/Housing.csv")

dummies=pd.get_dummies(df)

dummies=dummies.drop(["driveway_no","recroom_no","fullbase_no","gashw_no","airco_no","prefarea_no"],axis=1)

y=dummies.price

df_dummies=dummies

x=dummies.drop(["price"],axis=1)
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(xTrain,yTrain)
y_predict=model.predict(xTest)
model.score(xTest,yTest)
from sklearn.metrics import mean_squared_error 

MSE=mean_squared_error(yTest,y_predict)



from math import sqrt

RMSE=sqrt(MSE)

RMSE

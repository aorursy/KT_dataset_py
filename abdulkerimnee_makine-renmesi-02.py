import pandas as pd
from sklearn.model_selection import train_test_split
df=pd.read_excel("../input/real-estate-valuation-dataset/Real estate valuation data set.xlsx")
df.head()
df=df.rename(columns={"X1 transaction date":"İşlem Tarihi",
                  "X2 house age":"Ev Yaşı",
                  "X3 distance to the nearest MRT station":"Metroya Olan Uzaklık",
                  "X4 number of convenience stores":"Civardaki Market Sayısı",
                  "X5 latitude":"Enlem",
                  "X6 longitude":"Boylam",
                  "Y house price of unit area":"Birim Alan Fiyatı"})
df.head()
X = df.drop(['Birim Alan Fiyatı'],axis=1)#bağımsız değişkenler
y = df[["Birim Alan Fiyatı"]]#bağımlı değişken
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train,y_train)
model
model.intercept_#β0
model.coef_#β1
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
MSE=mean_squared_error(y_test,model.predict(X_test))
MSE
RMSE=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
RMSE
from sklearn.model_selection import cross_val_score
cross_val_score(model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")
#cv mse
np.mean(-cross_val_score(model,X_train,y_train,cv=10,scoring="neg_mean_squared_error"))
#cv rmse
np.sqrt(np.mean(-cross_val_score(model,X_train,y_train,cv=10,scoring="neg_mean_squared_error")))
r2_score(y_test, model.predict(X_test))
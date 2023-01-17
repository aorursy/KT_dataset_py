import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf 
from math import sqrt
df = pd.read_csv("/kaggle/input/sefaproje4/Zeki.csv") #data import etme
df.drop("Gunler", axis = 1, inplace = True)#günler bilgi vermiyor çıkarttım
X_train = df.iloc[0:25, 0:2]# X train = ilk 25 değer (X1 ve X2)
X_test = df.iloc[25:31, 0:2]# X test = son 5 değer (X1 ve X2)
y_train = df.iloc[0:25, 2:3]# y train = ilk 25 değer (Y)
y_test = df.iloc[25:31, 2:3]# y test = son 5 değer (Y)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
r2 = regressor.score(X_train,y_train)
print("R Kare değeri:",r2)
y_train = regressor.predict(X_test)
print("Tahmin Değerleri\n",y_train)
standartHata=mean_squared_error(y_test,y_train)
print("MSE Değeri:",standartHata)
print("RMSE Değeri:",sqrt(standartHata))
import pandas as pd

import numpy as np

import seaborn as sns 
d = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
d.info()
d["date"].value_counts()
d = d.drop(columns=["date","id"])
d.corrwith(d["price"])
x = d.drop(columns="price")
y = d["price"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
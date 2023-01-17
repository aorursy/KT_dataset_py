import pandas as pd

import seaborn as sns

import numpy as np
d = pd.read_csv("../input/insurance/insurance.csv")
d.info()
from sklearn.preprocessing import LabelEncoder

e = LabelEncoder()
d["sex"] = e.fit_transform(d["sex"])

d["smoker"] = e.fit_transform(d["smoker"])

d["region"] = e.fit_transform(d["region"])
d.info()
d.corrwith(d["charges"])
sns.heatmap(d.corr())
x = d.drop(columns="charges",axis=1)

y = d["charges"]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_test,y_test)

y_pred = model.predict(x_test)
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(x_test,y_test)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
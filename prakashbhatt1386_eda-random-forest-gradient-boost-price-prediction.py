import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
car = pd.read_csv("../input/usa-cers-dataset/USA_cars_datasets.csv")
car.head()
car.drop("Unnamed: 0",axis=1)
car.isna().sum()
car["brand"].value_counts().plot()
maxprice=car.groupby("brand")

maxprice=maxprice["model","price"].max()

maxprice=maxprice.sort_values(by="price",ascending=False)

maxprice.plot(kind="bar")
carmodel = car["model"].value_counts().plot()

car["year"].value_counts().plot(kind="bar")
carmodel=car.groupby("year")

carmodel=carmodel["brand","price"].max()

carmodel=carmodel.sort_values(by="year",ascending=False)

carmodel

car["title_status"].value_counts().plot(kind="bar")
maxmileage=car.groupby("brand")

maxmileage=maxmileage["model","mileage"].max()

maxmileage=maxmileage.sort_values(by="mileage",ascending=False)

maxmileage
car["mileage"].mean()
car["color"].value_counts().plot()
car_state=car.groupby("country")

car_state=car_state["state","brand","year","price"].max()

car_state



car["country"].value_counts().plot(kind="bar")
country=car.groupby("country")

country=country["state"].max()

country
car.columns
car.head()
car=car.drop(["Unnamed: 0","year","vin","lot","condition"],axis=1)
car.columns
x=car.drop("price",axis=1)

y=car["price"]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x = x.apply(le.fit_transform)
x.head()


from sklearn.model_selection import train_test_split,GridSearchCV

x_train,x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=42)
# Applying Stochastic gradient descent regressor techinique 
from sklearn.linear_model import SGDRegressor
lin_model=SGDRegressor()
param_grid = {

    'alpha': 10.0 ** -np.arange(1, 7),

    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],

    'penalty': ['l2', 'l1', 'elasticnet'],

    'learning_rate': ['constant', 'optimal', 'invscaling'],

}
sgd = GridSearchCV(lin_model, param_grid)
import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)

warnings.filterwarnings("ignore",category=FutureWarning)
sgd.fit(x_train, y_train)
y_pred=sgd.predict(x_test)
from sklearn.metrics import r2_score
r2=-(r2_score(y_test,y_pred))
r2
from sklearn import ensemble

from sklearn.ensemble import GradientBoostingRegressor

model = ensemble.GradientBoostingRegressor()

model.fit(x_train, y_train)
y_pred2=model.predict(x_test)
r2=r2_score(y_test,y_pred2)
r2
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(random_state=42)

forest_reg.fit(x_train, y_train)
y_pred3=forest_reg.predict(x_test)
r2=r2_score(y_test,y_pred3)
r2
# At last will apply simple linear regression technique 
#creat linear regression

from sklearn.linear_model import LinearRegression

lin_model=LinearRegression()

lin_model.fit(x_train,y_train)
y_pred4=lin_model.predict(x_test)
r2=r2_score(y_test,y_pred4)


r2

y_predection=forest_reg.predict(x)
y_predection
new_price = pd.DataFrame(y_predection)

new_price
car["predict_price"]=new_price
car.head()
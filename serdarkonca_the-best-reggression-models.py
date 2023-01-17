!pip install xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from warnings import filterwarnings
filterwarnings("ignore")
df = pd.read_csv("../input/kc-housesales-data/kc_house_data.csv")
df.head()
df.info()
df = df.dropna()
df.drop(["id","date"], axis = 1, inplace = True)
def compML(df, y, alg):
    
    y = df.price.values.reshape(-1,1)
    x = df.drop(["price"], axis = 1)
    
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 0, shuffle=True)
    
    model = alg().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    r2 = r2_score (y_test, y_pred)
    
    model_ismi = alg.__name__
    
    print(model_ismi, "R2_Score ---> ", r2)
models = [LinearRegression,
          DecisionTreeRegressor, 
          KNeighborsRegressor, 
          MLPRegressor, 
          RandomForestRegressor, 
          GradientBoostingRegressor,
          SVR,
          XGBRegressor]
for i in models:
    compML(df, "price", i)
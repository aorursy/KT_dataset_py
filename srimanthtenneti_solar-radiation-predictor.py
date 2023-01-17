import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
dataset = pd.read_csv("../input/SolarEnergy/SolarPrediction.csv")
dataset
temp = np.array(dataset["Temperature"]).reshape(-1,1)
press = np.array(dataset["Pressure"]).reshape(-1,1)
wd = np.array(dataset["WindDirection(Degrees)"]).reshape(-1,1)
humd = np.array(dataset["Humidity"]).reshape(-1,1)
speed = np.array(dataset["Speed"]).reshape(-1,1)
findata = np.concatenate((temp,press,humd,wd,speed),axis = 1)
findata.shape
dataset.shape
y = np.array(dataset["Radiation"])
regressor = RandomForestRegressor(n_estimators = 1200 , random_state = 9 , min_weight_fraction_leaf = 0.00001)
regressor.fit(findata,y)
plt.plot(y)
plt.plot(regressor.predict(findata))
regressor.score(findata,y)
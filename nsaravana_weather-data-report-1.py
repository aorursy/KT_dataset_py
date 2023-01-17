import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("../input/daily_weather.csv")
data.head()
data.shape
data.info()
data.shape
x=data["air_temp_9am"]
y=data["avg_wind_speed_9am"]
x.head()

y.head()
plt.scatter(x,y)
plt.show()
import seaborn as sns
sns.regplot(x="air_temp_9am",y="avg_wind_speed_9am",data=data)
plt.show()
sns.regplot(x="air_pressure_9am",y="avg_wind_speed_9am",data=data)
plt.show()
data=data.dropna(how="any",axis=0)
data.head()

sns.jointplot(x="avg_wind_speed_9am",y="relative_humidity_3pm",kind="reg",data=data)
plt.show()
data.columns

data.drop("number",axis=1)
x=np.array(data[["air_pressure_9am"]])
y=np.array(data[["relative_humidity_3pm"]])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model=LinearRegression()
model.fit(x_train,y_train)

sns.pairplot(data)
plt.show()
data.head()
sns.regplot(x="air_pressure_9am",y="relative_humidity_3pm",data=data)
plt.show()
sns.distplot(data['relative_humidity_3pm'])
plt.show()
data.corr()
data.drop("number",axis=1)
data.head()
fig = plt.figure(figsize = (20,11))
sns.heatmap(data.corr(), annot = True,cmap = "coolwarm")
plt.show()
corr=data.corr()
corr.nlargest(15,'relative_humidity_3pm')["relative_humidity_3pm"]
X = data[['avg_wind_direction_9am']]
Y = data[['relative_humidity_3pm']]

model.score(x_train,y_train)
predict = model.predict(x_test)
predict
from sklearn.model_selection import cross_val_score
score=cross_val_score(model,x,y,cv=2)
score


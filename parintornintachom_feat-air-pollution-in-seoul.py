import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')
df.head()

df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')
df.head()
cols =['SO2','NO2','CO','PM10']
sns.pairplot(data=df[cols])
df = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')
df.head()
cols =['SO2','NO2','CO','PM10']
x=df.CO.values.reshape(-1,1)
y=df.PM10.values.reshape(-1,1)
sns.lmplot(x='CO',y='PM10',data=df)
model=LinearRegression()
model.fit(x,y)
model.coef_,model.intercept_
x_input=[[600]]
y_predict=model.predict(x_input)
y_predict
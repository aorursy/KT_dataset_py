import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
data = pd.read_csv("../input/environmental-sensor-data-132k/iot_telemetry_data.csv")
data.head()
data.replace(['b8:27:eb:bf:9d:51', '00:0f:00:70:91:0a', '1c:bf:ce:15:ec:4d'], ['C1','C2','C3'], inplace=True)
data.head()
data['time'] = pd.to_datetime(data['ts'])
data['time']
data_1 = data[data.device == 'C1']
data_2 = data[data.device == 'C2']
data_3 = data[data.device == 'C3']

data_1
data_2
data_3
plt.plot(data_1['time'], data_1['co'], label='Device_C1')
plt.plot(data_2['time'], data_2['co'], label='Device_C2')
plt.plot(data_3['time'], data_3['co'], label='Device_C3')
plt.legend()
plt.show()
plt.plot(data_1['time'], data_1['humidity'], label='Device_C1')
plt.plot(data_2['time'], data_2['humidity'],label='Device_C2')
plt.plot(data_3['time'], data_3['humidity'],label='Device_C3')
plt.legend()
plt.show()
# Machine learning model between smoke and carbon Monoxide 
co_C1 = data_1.drop(['ts','time','device',"humidity","light", "lpg", "motion", "smoke", "temp"], axis=1)
co_S1 = data_1.drop(['ts','time','device',"humidity","light", "lpg", "motion", "temp", "co"], axis=1)
#co_C1
#co_T1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, Y_train, Y_test = train_test_split(co_C1, co_S1, test_size=0.2, random_state=0)
Linear = LinearRegression()
Linear.fit(X_train, Y_train)
print(Linear.score(X_train, Y_train))
y_pred = Linear.predict(X_test)
plt.plot(Y_test, y_pred)
plt.show()
#y = Linear.predict(new_data4)

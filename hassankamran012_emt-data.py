# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
electric_motor_temprature_data = pd.read_csv('../input/pmsm_temperature_data.csv')
electric_motor_temprature_data.describe()
electric_motor_temprature_data.info()
electric_motor_temprature_data.head()
u_q = electric_motor_temprature_data['u_q']

u_d = electric_motor_temprature_data['u_d']

i_q = electric_motor_temprature_data['i_q']

i_d = electric_motor_temprature_data['i_d']

torque = electric_motor_temprature_data['torque']
normalized_u_q = (u_q - u_q.min())/(u_q.max()-u_q.min())

normalized_u_d = (u_d - u_d.min())/(u_d.max()-u_d.min())

normalized_i_q = (i_q - i_q.min())/(i_q.max()-i_q.min())

normalized_i_d = (i_d - i_d.min())/(i_d.max()-i_d.min())

normalized_torque = (torque - torque.min())/(torque.max()-torque.min())
fig = plt.figure(figsize=(20,20))

ax = plt.axes(projection="3d")

ax.scatter3D(normalized_u_q, normalized_u_d, normalized_torque, s=0.5, c=normalized_torque, cmap=plt.get_cmap("jet"))



plt.show()
X = pd.concat([normalized_u_d, normalized_u_q, normalized_i_d, normalized_i_q], axis=1)
electric_motor_temprature_data.hist(bins=50, figsize=(20,15));
corr_matrix = electric_motor_temprature_data.corr()

corr_matrix["torque"].sort_values(ascending=False)

fig = plt.figure(figsize=(20,20))

ax = plt.axes(projection="3d")

ax.scatter3D(normalized_i_q, normalized_u_d, normalized_torque, s=0.5, c=normalized_torque, cmap=plt.get_cmap("jet"))



plt.show()
fig = plt.figure(figsize=(20,20))

ax = plt.axes(projection="3d")

ax.scatter3D(normalized_i_q, normalized_i_d, normalized_torque, s=0.5, c=normalized_torque, cmap=plt.get_cmap("jet"))



plt.show()
X.shape
normalized_torque.shape
normalized_torque = normalized_torque.to_dense()
normalized_torque.shape
normalized_torque = normalized_torque.values.reshape(1, 998070)
X = X.values.reshape(4, 998070)
X.shape
type(X)
normalized_torque.shape
type(normalized_torque)
X
normalized_torque
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.T, normalized_torque.T, test_size=0.1, random_state=42)
reg = LinearRegression().fit(X_train, y_train)
y_predict = reg.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_predict)
motor_speed = electric_motor_temprature_data['motor_speed']
normalized_motor_speed = (motor_speed - motor_speed.min())/(motor_speed.max() - motor_speed.min())
normalized_motor_speed = normalized_motor_speed.values.reshape(1, 998070)
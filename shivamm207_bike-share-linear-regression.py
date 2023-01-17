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
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np



data_frame = pd.read_csv('../input/bike_share.csv')

data_frame.head()
holidays_1 = data_frame['holiday'].values

working_days_1 = data_frame['workingday'].values



weekend_1 = []

for i in range(len(holidays_1)):

    if working_days_1[i] == 0 and holidays_1[i] == 0:

        weekend_1.append(1)

    else:

        weekend_1.append(0)



data_frame['weekend'] = weekend_1

data_frame.head()
data_frame_1 = data_frame.drop(columns=["registered", "count"])

data_frame_2 = data_frame.drop(columns=["casual", "count"])
pd.set_option('display.max_columns', None)

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)

print("Co-relation coefficient for casual bikes")

print(data_frame_1.corr())

print("Co-relation coefficient for registered bikes")

print(data_frame_2.corr())
plt.scatter(data_frame_1['weekend'], data_frame_1['casual'])

plt.xlabel('Weekend')

plt.ylabel('Number of casual bikes')

plt.show()
plt.scatter(data_frame_1['humidity'], data_frame_1['casual'], c='r')

plt.xlabel('Humidity')

plt.ylabel('Number of casual bikes')

plt.show()
plt.scatter(data_frame_1['atemp'], data_frame_1['casual'], c='r')

plt.xlabel('feel_temperature')

plt.ylabel('Number of casual bikes')

plt.show()
plt.scatter(data_frame_1['workingday'], data_frame_1['casual'])

plt.xlabel('working_day')

plt.ylabel('Number of casual bikes')

plt.show()

plt.scatter(data_frame_2['humidity'], data_frame_2['registered'])

plt.xlabel('humidity')

plt.ylabel('Number of registered bikes')

plt.show()
plt.scatter(data_frame_2['atemp'], data_frame_2['registered'])

plt.xlabel('feel_temperature')

plt.ylabel('Number of registered bikes')

plt.show()
x1 = data_frame_1[['humidity', 'weekend', 'atemp', 'workingday']].values

y1 = data_frame_1['casual'].values



x2 = data_frame_1[['humidity', 'atemp']].values

y2 = data_frame_2['registered'].values



train_x1, test_x1, train_y1, test_y1 = train_test_split(x1, y1, test_size=0.2, random_state=4)

train_x2, test_x2, train_y2, test_y2 = train_test_split(x2, y2, test_size=0.2, random_state=4)
model_casual = linear_model.LinearRegression()

model_casual.fit(train_x1, train_y1)



model_registered = linear_model.LinearRegression()

model_registered.fit(train_x2, train_y2)
predict_casual = model_casual.predict(test_x1)

predict_registered = model_registered.predict(test_x2)
print("For number of casual bikes")

test_MSE_1 = mean_squared_error(test_y1, predict_casual)

test_MAE_1 = mean_absolute_error(test_y1, predict_casual)

r2_value_1 = r2_score(test_y1, predict_casual)

print('test_MSE', test_MSE_1, 'test_MAE', test_MAE_1, 'r2_score', r2_value_1)
plot_y_1 = np.vstack((test_y1, predict_casual))

plot_y_1 = plot_y_1.T

plot_y_1 = plot_y_1[np.argsort(plot_y_1[:, 1])]

plot_x_1 = [i for i in range(len(test_y1))]

plt.scatter(plot_x_1, plot_y_1[:, 0], c='r', label='true_value')

plt.plot(plot_x_1, plot_y_1[:, 1], c='b', label='predicted_value')

plt.show()
print("For number of registered bikes")

test_MSE_2 = mean_squared_error(test_y2, predict_registered)

test_MAE_2 = mean_absolute_error(test_y2, predict_registered)

r2_value_2 = r2_score(test_y2, predict_registered)

print('test_MSE', test_MSE_2, 'test_MAE', test_MAE_2, 'r2_score', r2_value_2)
plot_y_2 = np.vstack((test_y2, predict_registered))

plot_y_2 = plot_y_2.T

plot_y_2 = plot_y_2[np.argsort(plot_y_2[:, 1])]

plot_x_2 = [i for i in range(len(test_y2))]

plt.scatter(plot_x_2, plot_y_2[:, 0], c='r', label='true_value')

plt.plot(plot_x_2, plot_y_2[:, 1], c='b', label='predicted_value')

plt.show()
y_true_f = np.add(test_y1, test_y2)

y_pred_f = np.add(predict_registered, predict_casual)



MAE = mean_absolute_error(y_true_f, y_pred_f)

MSE = mean_squared_error(y_true_f, y_pred_f)

print("Mean absolute error", MAE)

print("Mean Squared error", MSE)
plot_y = np.vstack((y_true_f, y_pred_f))

plot_y = plot_y.T

plot_y = plot_y[np.argsort(plot_y[:, 1])]

plot_x = [i for i in range(len(y_true_f))]

plt.scatter(plot_x, plot_y[:, 0], c='r', label='true_value')

plt.plot(plot_x, plot_y[:, 1], c='b', label='predicted_value')

plt.show()
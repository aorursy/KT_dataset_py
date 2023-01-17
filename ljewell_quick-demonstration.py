import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline 

matplotlib.style.use('ggplot')



data = pd.read_csv('../input/data.csv')

data.head(n=5)
plt.title('Estimated (blue) vs real (red) path')

plt.xlabel('x-coordinates')

plt.ylabel('y-coordinates')

plt.xlim(0,170) 

plt.ylim(120,0)

plt.scatter(x=data['x'], y=data['y'], c='b', s=data['cf']*10, alpha=0.1, label='Real')

plt.plot(data['realx'], data['realy'], '.r-', alpha=0.5)

plt.plot(data['x'], data['y'], '.b-', alpha=0.5)



plt.show()
# Calc distances and Mean Square Error (MSE)

import math

import numpy as np



def mse(estimate_a, a):

    return np.mean((estimate_a - a)**2)



def dist_xy(x, y):

    dist = 0

    prev_x = 0

    prev_y = 0

    for i in range(len(x)):

        if prev_x != 0:

            dist += math.sqrt((x[i] - prev_x)**2 + (y[i] - prev_y)**2)

        prev_x = x[i]

        prev_y = y[i]

    return dist



# Sort the dataframe by time

data = data.sort_values(by='time')



# Divide by 3.28 to get metres

estimated_dist = dist_xy(data['x'], data['y'])/3.28

real_dist = dist_xy(data['realx'], data['realy'])/3.28

percent = (estimated_dist/real_dist)*100

mse_x = mse(data['x'], data['realx'])

stand_dev_x = math.sqrt(mse_x)

mse_y = mse(data['y'], data['realy'])

stand_dev_y = math.sqrt(mse_y)



print("Estimated distance: {} metres".format(round(estimated_dist,2)))

print("Real distance: {} metres".format(round(real_dist,2)))

print("Percentage difference: {}%".format(round(percent,2)))

print("Means Square Error X: {}".format(round(mse_x,2)))

print("Mean Square Error Y: {}".format(round(mse_y,2)))

print("Standard Deviation X: {}".format(round(stand_dev_x,2)))

print("Standard Deviation Y: {}".format(round(stand_dev_y,2)))
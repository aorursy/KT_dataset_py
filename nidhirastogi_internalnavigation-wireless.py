# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('../input/data.csv')
data.head(10)

# Any results you write to the current directory are saved as output.
data.describe()
plt.title('Estimated (blue) vs. Real (red) path')
plt.xlabel('X-coord')
plt.ylabel('Y-coord')
plt.xlim(0,180) 
plt.ylim(-20,130)
plt.scatter(x=data['x'], y=data['y'], c='b', s=data['cf']*10, alpha=0.1, label='Real')
plt.plot(data['realx'], data['realy'], '.r-')
plt.plot(data['x'], data['y'], '.b-')

plt.show()
# Calc distances and Mean Square Error (MSE)

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

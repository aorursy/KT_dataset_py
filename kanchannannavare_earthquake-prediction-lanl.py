# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





%matplotlib inline



import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# idx [1..14] of the earthquake you'd like to animate

# first (0) and last (15) are note full cycles!!!

EARTHQUAKE = 1



# Datapoints inside the window; Set this lower if you'd like to zoom in.

WINDOW_SIZE = 150000



# Window step size

STEP_SIZE = WINDOW_SIZE // 5



# Refresh interval; lower=faster animation

REFRESH_INTERVAL = 100



earthquakes = [5656574, 50085878, 104677356, 138772453, 187641820, 218652630, 245829585, 307838917,

               338276287, 375377848, 419368880, 461811623, 495800225, 528777115, 585568144, 621985673]



train_df = pd.read_csv('../input/train.csv', nrows=earthquakes[EARTHQUAKE + 1] - earthquakes[EARTHQUAKE],

                       skiprows = earthquakes[EARTHQUAKE] + 1,

                       names=['acoustic_data', 'ttf'])



train_df.tail(20)

# data = train_df['acoustic_data'].value_counts()

# data.plot(kind="bar")





split_train_df = train_df.head(1000000)

print("data size", train_df.size)

X = split_train_df['acoustic_data']

Y = split_train_df['ttf']

plt.scatter(X, Y)

plt.show()
from scipy import stats



slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)



print(r_value ** 2)
def predict(x):

    return slope * x + intercept



fitLine = predict(X)



plt.scatter(X, Y)

plt.plot(X, fitLine, c='r')

plt.show()
from scipy import optimize

def test_func(x, a, b):

    return a * np.sin(b * x)



params, params_covariance = optimize.curve_fit(test_func, X, Y,

                                               p0=[2, 2])



print(params)



plt.figure(figsize=(6, 4))

plt.scatter(X, Y, label='Data')

plt.plot(X, test_func(X, params[0], params[1]),

         label='Fitted function', color="r")



plt.legend(loc='best')



plt.show()
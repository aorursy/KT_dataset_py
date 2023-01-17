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
x_train = [] 

x_test = []

y_train = []

y_test = [] 

l = 40#length of train data
for i in range(1,l+1):

    x_train.append(4*(i-1)/(l-1)-2)

for i in range(1, l):

    x_test.append(4*(i-0.5)/(l-1)-2)

for i in x_train:

    y_train.append(1/(1+25*i*i))

for i in x_test:

    y_test.append(1/(1+25*i*i))
import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(20,10))

plt.plot(x_train, y_train, 'ro')

plt.plot(x_test, y_test, 'bs')

plt.gca().legend(('train values','test values'))

plt.axis([-2.25, 2.25, -.1, 1.25])

plt.show()
import numpy as np

def squared_error(y_true, predictions):

    y_true = np.array(y_true)

    predictions = np.array(predictions)

    mse = np.mean((y_true - predictions)**2)

    return mse
def getNeighbours(x,y, elem):

    if elem <= x[0]:

        return [(x[0],y[0]),(x[1],y[1])]

    if elem >= x[len(x)-1]:

        return [(x[len(x)-1], y[len(y)-1]),(x[len(x)-2],y[len(y)-2])]

    else:

        for i in range(len(x)):

            if elem <= x[i]:

                return [(x[i-1],y[i-1]),(x[i],y[i])]



def calcY(x,y,el):

    x1 = getNeighbours(x,y,el)[0][0]

    x2 = getNeighbours(x,y,el)[1][0]

    y1 = getNeighbours(x,y,el)[0][1]

    y2 = getNeighbours(x,y,el)[1][1]

    return y1 + (y2-y1)*(el-x1)/(x2-x1)



def func(x,y,xk):

    arr = []

    for i in range(len(xk)):

        arr.append(calcY(x,y,xk[i]))

    return arr



prediction = func(x_train,y_train,x_test)

print("Error: {}".format(squared_error(y_test, prediction)))

plt.figure(figsize=(15,10))

plt.plot(x_train,y_train,'ko-') # train

plt.plot(x_test, y_test,'gs' ) # test

plt.plot(x_test, prediction, 'bo') # predicted

plt.gca().legend(('train values','test values', 'predicted'))

plt.axis([-2.25, 2.25, -0.25, 1.25])

plt.show()
from scipy.interpolate import interp1d

f2 = interp1d(x_train, y_train, kind='cubic')



from scipy.interpolate import make_interp_spline, BSpline

xnew = np.linspace(min(x_train),max(x_train), 400)

spl = make_interp_spline(x_train, f2(x_train), k=3) #BSpline object

power_smooth = spl(xnew)



xnew_test = np.linspace(min(x_test),max(x_test), l-1)

spl = make_interp_spline(x_test, f2(x_test), k=3) #BSpline object

power_smooth_test = spl(xnew_test)

print("Error: {}".format(squared_error(y_test, power_smooth_test)))



plt.figure(figsize=(50,30))

plt.plot(xnew, power_smooth, 'k-') # cubic spline

plt.plot(x_train, y_train, 'ko') # train

plt.plot(x_test, y_test,'gs') # test

plt.plot(xnew_test, power_smooth_test, 'bo')

plt.gca().legend(('train curve','train values', 'test values', 'predicted'))

plt.axis([-2.25, 2.25, -0.25, 1.25])

plt.show()
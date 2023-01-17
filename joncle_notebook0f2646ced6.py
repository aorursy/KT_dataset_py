import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/test.csv.zip')



data = data[data.MISSING_DATA == False]

data.head()
#data.POLYLINE.apply(lambda x: '0' if len(x) < 2 else x) 

#data = data[data.POLYLINE != '0']

# data = data[data.MISSING_DATA != True]

def extract_data(x):

    splitted = x[2:-2].split('],[')      

    time_slots = len(splitted)

    if len(splitted) > 0 and len(splitted[0].split(',')) > 1 :

        x,y = splitted[0].split(',')

        return [x, y, time_slots]

    else :

        return '0'



data['x_y_slots']= data['POLYLINE'].apply(extract_data)

data = data[data.x_y_slots != '0']

data['x'] = data['x_y_slots'].apply(lambda x: float(x[0]) if len(x) > 0 else NULL)

data['y'] = data['x_y_slots'].apply(lambda x: float(x[1]) if len(x) > 0 else NULL)

data['time_slots'] = data['x_y_slots'].apply(lambda x: x[2])

data.tail()
import matplotlib.pyplot as plt 



plt.scatter(data.x, data.y, data.time_slots)

plt.xlabel('x')

plt.ylabel('y')

plt.show()



plt.scatter(data.x, data.time_slots)

plt.xlabel('x')

plt.ylabel('time')

plt.show()



plt.scatter(data.y, data.time_slots)

plt.xlabel('y')

plt.ylabel('time')

plt.show()
from mpl_toolkits.mplot3d import Axes3D

# Axes3D.scatter(data.x, data.y, data.time_slots)

fig = plt.figure().gca(projection='3d')

fig.scatter(data.x, data.y, data.time_slots)

fig.set_xlabel('x')

fig.set_ylabel('y')

fig.set_zlabel('time_slot')

fig
import sklearn.cross_validation as cv

from sklearn.linear_model import LinearRegression



data1 = pd.DataFrame()

data1['x'] = data['x']

data1['y'] = data['y']
train_d, test_d, train_tar, test_tar = cv.train_test_split(data1, data.time_slots, test_size=0.2, random_state=5)

#from sklearn import linear_model

#lr = linear_model.SGDClassifier()

lr = LinearRegression(normalize=True)

lr.fit(train_d, train_tar)
lr.score(train_d, train_tar)
lr.score(test_d, test_tar)
from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=2)

poly_data = poly.fit_transform(data1)



train_d, test_d, train_tar, test_tar = cv.train_test_split(poly_data, data.time_slots, test_size=0.2, random_state=5)





lr = LinearRegression(normalize=False)

lr.fit(train_d, train_tar)



print ((lr.score(train_d, train_tar)), (lr.score(test_d, test_tar)))
data.time_slots = (data.time_slots - data.time_slots.mean()) / (data.time_slots.max() - data.time_slots.min())

plt.scatter(data.TIMESTAMP, data.time_slots)

plt.xlabel('timestamp')

plt.ylabel('time slots')

plt.show()

import math

sin_timestamp = data.TIMESTAMP.apply(lambda x: math.cos(x*2*3.14/(60*60*24))) 



plt.scatter(sin_timestamp, data.time_slots)

plt.xlabel('timestamp')

plt.ylabel('time slots')

plt.show()
data1['cos_ts'] = data.TIMESTAMP.apply(lambda x: math.cos(x*2*3.14/(60*60*24))) 

data1['sin_ts'] = data.TIMESTAMP.apply(lambda x: math.sin(x*2*3.14/(60*60*24))) 



# partitioning

train_d, test_d, train_tar, test_tar = cv.train_test_split(data1, data.time_slots, test_size=0.2, random_state=5)



lr = LinearRegression()

lr.fit(train_d, train_tar)

print ((lr.score(train_d, train_tar)), (lr.score(test_d, test_tar)))
def extract_end_point(x):

    splitted = x[2:-2].split('],[')

    x,y = splitted[-1].split(',')

    return [x, y]



data['x1_y1']= data['POLYLINE'].apply(extract_end_point)

data['x1'] = data['x1_y1'].apply(lambda x: float(x[0]))

data['y1'] = data['x1_y1'].apply(lambda x: float(x[1]))



# data1['x1'] = data['x1']

# data1['y1'] = data['y1']



train_d, test_d, train_tar, test_tar = cv.train_test_split(data1, data.time_slots, test_size=0.2, random_state=5)


lr = LinearRegression(normalize=True)

lr.fit(train_d, train_tar)

print ((lr.score(train_d, train_tar)), (lr.score(test_d, test_tar)))
from sklearn.preprocessing import PolynomialFeatures



poly = PolynomialFeatures(degree=2)

poly_data = poly.fit_transform(data1)



train_d, test_d, train_tar, test_tar = cv.train_test_split(poly_data, data.time_slots, test_size=0.2, random_state=5)





lr = LinearRegression(normalize=False)

lr.fit(train_d, train_tar)



print ((lr.score(train_d, train_tar)), (lr.score(test_d, test_tar)))
((test_tar - lr.predict(test_d))**2).mean()
dz = pd.DataFrame()

dz.x = data.x1 - data.x

dz.y = data.y1 - data.y



dz.x = dz.x.apply(lambda i: i*-1 if i<1 else i)

dz.y = dz.y.apply(lambda i: i*-1 if i<1 else i)

dz.z = dz.x + dz.y

dz.time = data.time_slots
plt.scatter(dz.x, dz.time)

plt.xlabel('x')

plt.ylabel('time')

plt.show()



plt.scatter(dz.y, dz.time)

plt.xlabel('y')

plt.ylabel('time')

plt.show()



plt.scatter(dz.z, dz.time)

plt.xlabel('total diff')

plt.ylabel('time')

plt.show()

## Using distance 
def distance_heuristic(x):

    splitted = x[2:-2].split('],[')

    time_slots = len(splitted)

    if len(splitted) > 1 and len(splitted[0].split(',')) > 1 :

        xdist = 0

        ydist = 0

        prevx, prevy = splitted[0].split(',')

        prevx,prevy = float(prevx), float(prevy)

        for point in splitted[1:]:

            x,y = point.split(',')

            x,y = float(x), float(y)

            xdist += math.fabs(prevx -x)

            ydist += math.fabs(prevy -y)

        return [xdist, ydist]

    else :

        return [0,0]
data['xdist_ydist'] = data.POLYLINE.apply(distance_heuristic)

data2 = pd.DataFrame()

data2['xdist'] = data.xdist_ydist.apply(lambda x: x[0])

data2['ydist'] = data.xdist_ydist.apply(lambda x: x[1])

data2['time'] = data.time_slots

data2 = data2[data2.xdist + data2.ydist < 50]
data2.dist = data2.xdist + data2.ydist 

plt.scatter(data2.dist, data2.time)

plt.xlabel('x')

plt.ylabel('time')

plt.show()
td = pd.DataFrame(data2.xdist)

td['ydist'] = data2.ydist

train_d, test_d, train_tar, test_tar = cv.train_test_split(td, data2.time, test_size=0.2, random_state=5)
lr = LinearRegression(normalize=True)

lr.fit(train_d, train_tar)

print ((lr.score(train_d, train_tar)), (lr.score(test_d, test_tar)))
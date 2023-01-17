"""
This project will require you to have a little practice with processing spatial and 
temporal data. Specifically, you are to use prediction to detect which “significant area” 
is likely to have highest yield in terms of a “value” from multiple time-series sources.

Description:
You are given an input-configuration which consists of:
- Collection of polygons {P1, P2, P3, …, Pn}
	o Each polygon Pi will be represented as a sequence of points 
      representing its vertices in a counter-clockwise order 
      (Note: no polygons with holes; no self-intersecting polygons).
	o Each vertex will be represented as a pair of (x,y) values, 
      indicating the coordinates in a suitable system.
	o The “terminator” of the sequence of points is indicated with the pair (-1000, -1000).
- Collection of locations {L1, L2, …, Lk}
	o Each location is represented as a pair of (x,y) values, indicating coordinates 
      in the same coordinate system as the values of the vertices of the polygon.
- Collection of time-series values for each location {TS1, TS2, …, TSk}
	o Each time series will be associated with a particular location (e.g., TSi is at Li).
	o Each will be represented as a sequence of values [v i1 , v i2 , …, v iN ]

The components of the input configuration will be given as csv files. 
The important additional formatting info is:
1. For the polygons, the delimiter of two coordinates pairs is " ; " and 
   EOF/EOL is denoted by "-1000, -1000".
2. For the locations (points), the delimiter between two points (i.e., the pairs 
   of coordinate values) is " ; "and EOF/EOL is denoted by " -1000, -1000".
3. For the time series, there will be multiple :
	a. the first column denotes the index and the last column is the EOL "-1000".
	b. the number of rows will match the number of locations. 
       Each particular index in the first column of the i-th row, will indicate 
       that the rest (the actual time-series data) is associated with the location Li.

For a given configuration, there are two additional inputs:
	- Q – a query-polygon (given as a sequence of coordinates of its vertices in 
          counter-clockwise order, separated by “;” and with “-1000, -1000” as EOL/EOF).
	- W – a time window (given as an integer).

Your job is to answer the following query:
Which area among all the intersections of Q with each of the P1, P2, …, Pn has the highest 
total sum of the predicted values in the time series inside that area for the window W.

You should use ARIMA for the prediction.



My Approach:
1. Find intersection with Q
2. Build models of locations in Q
3. Predict values using models of window W
4. Sum values which is the max


# assumption
- Input w is the number of steps from the last step, which is used for prediction
- sum of the predicted values in the ime series is sum of all steps in all of the target locations
- the non intersected areas with P1, P2, …, Pn are not part of result
- each time series has different params for ARIMA

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from shapely.geometry import Point, Polygon
import descartes
import matplotlib.pylab as plt
%matplotlib inline

import os
print(os.listdir("../input"))
# we are gonna use shapely to analyze the spatial data

with open('../input/polygons.csv') as f:
    polygons_raw = f.read()
polygons_raw = polygons_raw.replace(' ', '')
polygons_raw
with open('../input/points.csv') as f:
    points_raw = f.read()
points_raw = points_raw.replace(' ', '')
points_raw
# parse points

#points_x = []
#points_y = []
points = []
for point in points_raw.split('-1000,-1000')[0].split(';'):
    if point == '':
        continue
    point = point.split(',')
    points.append(Point(float(point[0].lstrip()), float(point[1].lstrip())))
points[:3]
# display points

for point in points:
    plt.plot(point.x, point.y, 'ro')
plt.show()
# parse polygons

polygons = []
for polystr in polygons_raw.replace('\n', '').split('-1000,-1000'):
    if polystr == '':
        continue
    polygon = []
    for vertex in polystr.split(';'):
        if vertex == '':
            continue
        vertex = vertex.split(',')
        vertex[0] = float(vertex[0])
        vertex[1] = float(vertex[1])
        polygon.append(vertex)
    print(polygon)
    polygon = Polygon(polygon)
    polygons.append(polygon)
# display polygons + points

print('# points: {}'.format(len(points)))
print('# polygons: {}'.format(len(polygons)))

fig, ax = plt.subplots()

for point in points:
    ax.plot(point.x, point.y, 'ro')
for polygon in polygons:
    ax.add_patch(descartes.PolygonPatch(polygon, fc='blue', alpha=0.5))
ax.axis('equal')

plt.show()
# some example input

q = Polygon([(96,59), (96,52), (102,52), (102,58)])
w = 10
# find intersections with q

intersections = []

for polygon in polygons:
    intersection = polygon.intersection(q)
    if not intersection.is_empty:
        intersections.append(intersection)
intersections
targets = [] # target locations
for intersection in intersections:
    targets.append([])
    for i in range(len(points)):
        if not intersection.intersection(points[i]).is_empty:
            targets[-1].append(i)
targets
# now show the target regions
# yellow region = q
# green regions = groups of locations we need to apply arima individually

fig, ax = plt.subplots()

for point in points:
    ax.plot(point.x, point.y, 'ro')
for polygon in polygons:
    ax.add_patch(descartes.PolygonPatch(polygon, fc='blue', alpha=0.5))

ax.add_patch(descartes.PolygonPatch(q, fc='yellow', alpha=0.5))
for intersection in intersections:
    ax.add_patch(descartes.PolygonPatch(intersection, fc='green', alpha=0.5))
    
for target in targets:
    for point_idx in target:
        ax.plot(points[point_idx].x, points[point_idx].y, 'bo')
    
ax.axis('equal')

plt.show()
targets
# Now, we are gonna deal with ts data for each of the target locations
# i followed https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/ to apply ARIMA predictions

with open('../input/ts.csv') as f:
    ts_raw = f.read()
ts_raw = ts_raw.replace(' ', '')
ts_raw[:100]
# parse time series 

tss = [] # list of time series
for ts_tmp in ts_raw.replace('\n', '').split('-1000'):
    if ts_tmp == '':
        continue
    ts = []
    for t in ts_tmp.split(','):
        if t == '':
            continue
        ts.append(int(t))
    tss.append(ts)
print(len(tss[0]))
print(tss[0])
print(len(tss[1]))
print(tss[1])
# Display a couple of time series

plt.plot(tss[0])
plt.show()
plt.plot(tss[56])
plt.show()
# we are gonna see ACF and PACF plots to determine q and p parameters for ARIMA
from statsmodels.tsa.stattools import acf, pacf

point_idx = 0

# lets see first 3
for point_idx in range(3):
    lag_acf = acf(tss[point_idx], nlags=20)
    lag_pacf = pacf(tss[point_idx], nlags=20, method='ols')
    
    lower_confidence = -1.96/np.sqrt(len(tss[point_idx]))
    upper_confidence = 1.96/np.sqrt(len(tss[point_idx]))

    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=lower_confidence,linestyle='--',color='gray')
    plt.axhline(y=upper_confidence,linestyle='--',color='gray') # upper confidence
    plt.title('{} Autocorrelation Function'.format(point_idx))

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=lower_confidence,linestyle='--',color='gray')
    plt.axhline(y=upper_confidence,linestyle='--',color='gray')
    plt.title('{} Partial Autocorrelation Function'.format(point_idx))
    plt.tight_layout()
    
    plt.show()
    
    # find p
    for i in range(20):
        if lag_acf[i] < upper_confidence:
            # find x value of intersection
            # y = ax+b
            x1, y1 = i-1, lag_acf[i-1]
            x2, y2 = i, lag_acf[i]
            a = (y1-y2)/(x1-x2)
            b = y1-a*x1
            x_intersect = (upper_confidence-b)/a
            print('acf intersects with upper confidence first at {}'.format(x_intersect))
            break
    # find q
    for i in range(20):
        if lag_pacf[i] < upper_confidence:
            # find x value of intersection
            # y = ax+b
            x1, y1 = i-1, lag_pacf[i-1]
            x2, y2 = i, lag_pacf[i]
            a = (y1-y2)/(x1-x2)
            b = y1-a*x1
            x_intersect = (upper_confidence-b)/a
            print('pacf intersects with upper confidence first at {}'.format(x_intersect))
            break
# apply arima for the first model
# turns out that p, q are chosen with no pattern (the one closest? - no, the one bigger? - no, etc)
# so, we actually need try and catch 2 * 2 = 4 times

from statsmodels.tsa.arima_model import ARIMA

# order = (q, d, p)
model = ARIMA(tss[56], order=(3, 0, 1)) 
model_fit = model.fit(disp=-1)  
plt.plot(tss[0])
plt.plot(model_fit.fittedvalues, color='red')
plt.show()
output = model_fit.forecast(w)
output

# returns
# forecast (array) – Array of out of sample forecasts
# stderr (array) – Array of the standard error of the forecasts.
# conf_int (array) – 2d array of the confidence interval for the forecast

# NOW PREDICT for first time series

plt.plot(tss[0])
plt.plot(model_fit.fittedvalues, color='red')
start = len(tss[0])
steps = []
for i in range(w):
    steps.append(start+i)
plt.plot(steps, output[0], color='purple')
plt.show()
# Now, since we only analyzed the first ts, we will make a function to automatically calculate the result for all targets.

sums = [] # sum for all target regions
for target in targets:
    tmp_sum = 0
    for point_idx in target:
        print(point_idx)
        
        """ calculate possible parameters """
        lag_acf = acf(tss[point_idx], nlags=20)
        lag_pacf = pacf(tss[point_idx], nlags=20, method='ols')
        lower_confidence = -1.96/np.sqrt(len(tss[point_idx]))
        upper_confidence = 1.96/np.sqrt(len(tss[point_idx]))
        
        p_max = 0
        q_max = 0
        
        for i in range(20):
            if lag_acf[i] < upper_confidence:
                p_max = i
                break
        for i in range(20):
            if lag_pacf[i] < upper_confidence:
                q_max = i
                break
        
        possible_params = [(p_max, 0, q_max), (p_max-1, 0, q_max), (p_max, 0, q_max-1), (p_max-1, 0, q_max-1)]
        
        """ apply ARIMA prediction for all possible parameters and choose the one that works """
        
        for param in possible_params:
            try:
                model = ARIMA(tss[point_idx], order=(2, 0, 2)) 
                model_fit = model.fit(disp=-1)
                output = model_fit.forecast(w)
                if np.isnan(output[0][0]):
                    continue
                for value in output[0]: # sum all values in predicted y
                    tmp_sum += value
                print('used {} for ts {}'.format(param, point_idx))
                break
            
            except Exception as e:
                continue
            
                    
    sums.append(tmp_sum)
sums
# now just find the max in the sums

print('input: q = {}, w = {}'.format(q, w))
print('answer: area {} with predicted sum of {}'.format(sums.index(max(sums)), max(sums)))
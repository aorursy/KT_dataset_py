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
# import data and support vector machine
from sklearn import svm
data = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')

# before drop missing values
before = len(data)
print('Before drop:', before)
# drop missing values
data.dropna(inplace=True)
# reset index so no holes in dataframe
data.reset_index(drop=True, inplace=True)
# after drop missing values
after = len(data)
print('After drop: ', after)
print('Dropped:    ', before - after, 'values or', round((before-after)/before*100, 2), 'percent of values')

# to see what column are named
print("Columns are named:\n", data.columns)
# Automatic generation of encoding mapping
i = 0
print("{", end='')
types = sorted(set(data.get('Primary Type')))
for type in types:
    i += 1
    if i%4 == 0:
        print("'%s':%d, " % (type, i), end='\n')
    elif i == len(types):
        print("'%s':%d" % (type, i), end='')
    else:
        print("'%s':%d, " % (type, i), end='')
print("}", end='')
# Encoding to ints for SVM
cleanup = {"Primary Type" : {'ARSON':1, 'ASSAULT':2, 'BATTERY':3, 'BURGLARY':4, 
'CONCEALED CARRY LICENSE VIOLATION':5, 'CRIM SEXUAL ASSAULT':6, 'CRIMINAL DAMAGE':7, 'CRIMINAL TRESPASS':8, 
'DECEPTIVE PRACTICE':9, 'GAMBLING':10, 'HOMICIDE':11, 'HUMAN TRAFFICKING':12, 
'INTERFERENCE WITH PUBLIC OFFICER':13, 'INTIMIDATION':14, 'KIDNAPPING':15, 'LIQUOR LAW VIOLATION':16, 
'MOTOR VEHICLE THEFT':17, 'NARCOTICS':18, 'NON - CRIMINAL':19, 'NON-CRIMINAL':20, 
'NON-CRIMINAL (SUBJECT SPECIFIED)':21, 'OBSCENITY':22, 'OFFENSE INVOLVING CHILDREN':23, 'OTHER NARCOTIC VIOLATION':24, 
'OTHER OFFENSE':25, 'PROSTITUTION':26, 'PUBLIC INDECENCY':27, 'PUBLIC PEACE VIOLATION':28, 
'ROBBERY':29, 'SEX OFFENSE':30, 'STALKING':31, 'THEFT':32, 
'WEAPONS VIOLATION':33}}
data.replace(cleanup, inplace=True)
# confirm encoding worked
print('Encoded Primary Type\n', data.head(10).get('Primary Type'))
##########################################################################################################
dataPlot = data.query("Arrest == True") # filter
print("Total  :", len(dataPlot))
dataPlot = dataPlot.sample(n=100000) # sample
print("Sampled:", len(dataPlot))
dataPlot.dropna(inplace=True) # drop missing values
longitude = dataPlot.get('Longitude')
latitude = dataPlot.get('Latitude')
dataframe = pd.DataFrame({'longitude': longitude, 'latitude': latitude}) # dataframe
##########################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
sns.kdeplot(dataframe.longitude, dataframe.latitude) # basic contour plot
# sns.kdeplot(dataframe.longitude, dataframe.latitude, cmap="Reds", shade=True, bw=.15) # density plot
# sns.kdeplot(dataframe.longitude, dataframe.latitude, cmap="Blues", shade=True, shade_lowest=True) # contour plot
axes = plt.gca() # change window focus
axes.set_xlim([-87.85, -87.50])
axes.set_ylim([41.60, 42.05])
##########################################################################################################
# rename column
data.rename(columns = {'Primary Type':'Primary'}, inplace=True)
# before
print('Before filtering')
print(data[0:10].get(['Primary', 'Location']))
# filter
data.query("Primary == 11 or Primary == 2 or Primary == 6 or Primary == 29", inplace=True) # violent crimes
data.dropna(inplace=True)
# change column name back
data.rename(columns = {'Primary':'Primary Type'}, inplace=True)
# after
print('After filtering')
print(data[0:10].get(['Primary Type', 'Location']))
# take small sample for SVM training
data = data.sample(n=50000)   # NUMBER OF SAMPLES
data.reset_index(drop=True, inplace=True)
print("Samples taken:", len(data))
print(data.get(['Primary Type', 'Longitude', 'Latitude'])[0:10])
# get Primary Type data as category
y = data.get('Arrest')   # response
print('  Primary Type')
print(y[0:10])
# circumvented problem of weird type error by joining other columns in dataset together to get location
longitude = data.get('Longitude')
latitude = data.get('Latitude')
# join to get one location value
x = []
# dec_vals = 4    # number of decimal places to keep
for i in range(len(data)):
    # x.append([round(longitude[i], dec_vals), round(latitude[i], dec_vals)])
    x.append([longitude[i], latitude[i]])
for i in range(len(x[0:10])):
    print(x[i])
# try to make into arrays
x = np.array(x)
y = np.array(y)
# see if any issues in input
print(x[0:10])
print(y[0:10])
# run Support Vector Machine where x is location and y is type of crime
model = svm.SVC(kernel='rbf', probability=True)
model.fit(x, y)
# run classification prediction
locations = []
dec = 1000
xMin = -87.954
xMax = -87.515
yMin = 41.648
yMax = 42.022
step = 0.001
for xx in range(int(xMin * dec), int(xMax * dec), int(step * dec)):
    for yy in range(int(yMin * dec), int(yMax * dec), int(step * dec)):
        locations.append([xx / dec, yy / dec])
"""
# alternative locations
locations = [[-87.608560, 41.891614],   # Navy Pier in City
             [-87.621708, 41.879486],   # Art Institute in City
             [-87.620996, 41.862419],   # Central Station in City
             [-87.632898, 41.851386],   # Chinatown
             [-87.600155, 41.788962],   # University of Chicago
             [-87.719158, 41.979565],   # Northeastern Illinois University
             [-87.627629, 41.834802],   # Illinois Institute of Technology
             [-87.689535, 41.939916],   # North side Police Station
             [-87.643642, 41.903055],   # City area Police Station
             [-87.660301, 41.779763]]   # South side Police Station
"""
predictions = model.predict_proba(locations)
print("Location is:\n", locations[0:10])
print("Prediction is:\n", predictions[0:10])
width = 0
for xx in range(int(xMin * dec), int(xMax * dec), int(step * dec)):
    width += 1
print("Width is", width)
height = len(locations) / width
print("Height is", height)
print("Area is", len(locations))
print("Number of support vectors used: ", len(model.support_vectors_))
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
# plot data and SVM lines
plot_decision_regions(x, y.astype(np.integer), clf=model, legend=2) ##########
# graph labels
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("SVM Result")
axes = plt.gca()
axes.set_xlim([-87.85, -87.50])
axes.set_ylim([41.60, 42.05])
plt.show()
rect_array = []
row = []
col = 0
for i in range(len(predictions)):
    row.append(predictions[i][0])
    col += 1
    if col == width:
        col = 0
        rect_array.append(row)
        row = []
print("Width is", len(rect_array[0]))
print("Height is", len(rect_array))
import seaborn as sns
graph = sns.heatmap(rect_array, square=True)

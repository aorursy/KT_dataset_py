# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#1
my_data = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')
#2
my_data.describe()

#3
my_data.isna().sum()
my_data.fillna(my_data.median(axis = 0), axis=0, inplace=True)
my_data.isna().sum()
my_data['country'].fillna(my_data['country'].mode().iloc[0], inplace=True)
my_data['designation'].fillna(my_data['designation'].mode().iloc[0], inplace=True)
my_data['province'].fillna(my_data['province'].mode().iloc[0], inplace=True)
my_data['region_1'].fillna(my_data['region_1'].mode().iloc[0], inplace=True)
my_data['region_2'].fillna(my_data['region_2'].mode().iloc[0], inplace=True)
my_data['taster_name'].fillna(my_data['taster_name'].mode().iloc[0], inplace=True)
my_data['taster_twitter_handle'].fillna(my_data['taster_twitter_handle'].mode().iloc[0], inplace=True)
my_data['variety'].fillna(my_data['variety'].mode().iloc[0], inplace=True)
my_data.isna().sum()

#6
from sklearn.model_selection import train_test_split

train, test = train_test_split(my_data, test_size=0.2)
#7
import random
import math
import pylab as pl
from matplotlib.colors import ListedColormap

def classifyKNN (trainData, testData, k, numberOfClasses):
    #Euclidean distance between 2-dimensional point
    def dist (a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    testLabels = []	
    for testPoint in testData:
        #Claculate distances between test point and all of the train points
        testDist = [ [dist(testPoint, trainData[i][0]), trainData[i][1]] for i in range(len(trainData))]
        #How many points of each class among nearest K
        stat = [0 for i in range(numberOfClasses)]
        for d in sorted(testDist)[0:k]:
            stat[d[1]] += 1
        #Assign a class with the most number of occurences among K nearest neighbours
        testLabels.append( sorted(zip(stat, range(numberOfClasses)), reverse=True)[0][1] )
    return testLabels

classifyKNN(train, test)
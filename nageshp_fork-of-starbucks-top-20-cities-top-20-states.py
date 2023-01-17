# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
fileName = '../input/directory.csv'

starbucksDF = pd.read_csv(fileName);
starbucksStoreByState = starbucksDF.groupby(['Country', 'State/Province']).count().reset_index().iloc[:,range(3)]

starbucksStoreByState["Country-State"] = starbucksStoreByState["Country"]+ "-" + starbucksStoreByState["State/Province"]

starbucksStoreByState.rename(columns={'Brand': 'StoreCount'}, inplace=True)

starbucksStoreByState.columns
starbucksStoreByStateOrderedTop20 = starbucksStoreByState.sort_values(['StoreCount'], ascending=[0]).head(20)
import matplotlib.pyplot as plt

import numpy as np



# The X axis can just be numbered 0,1,2,3...

x = np.arange(len(starbucksStoreByStateOrderedTop20["Country-State"]))



plt.bar(x, starbucksStoreByStateOrderedTop20["StoreCount"])

plt.xticks(x, starbucksStoreByStateOrderedTop20["Country-State"], rotation=90)



plt.figure(figsize=(2000,1000))
# California state has highest number

# England on #3
starbucksStoreByCityOrderedTop20 = starbucksDF.groupby(['Country', 'State/Province', 'City']).count().reset_index().sort_values(['Brand'], ascending=[0]).iloc[:,range(4)].head(20)

starbucksStoreByCityOrderedTop20["Country-State-City"] = starbucksStoreByCityOrderedTop20["Country"]+ "-" + starbucksStoreByCityOrderedTop20["State/Province"]+ "-" + starbucksStoreByCityOrderedTop20["City"]

starbucksStoreByCityOrderedTop20.rename(columns={'Brand': 'StoreCount'}, inplace=True)
import matplotlib.pyplot as plt

import numpy as np



# The X axis can just be numbered 0,1,2,3...

x = np.arange(len(starbucksStoreByCityOrderedTop20["Country-State-City"]))



plt.bar(x, starbucksStoreByCityOrderedTop20["StoreCount"])

plt.xticks(x, starbucksStoreByCityOrderedTop20["Country-State-City"], rotation=90)



plt.figure(figsize=(2,1))
# Interesting - Seoul city has more starbucks stores than New York city
from geopy.geocoders import Nominatim
geolocator = Nominatim()
from geopy.distance import vincenty

newport_ri = (41.49008, -71.312796)

cleveland_oh = (41.499498, -81.695391)

print(vincenty(newport_ri, (41.499498, -81.695391)).miles)
starbucksDF.columns
#from sklearn.preprocessing import Imputer



#imp = Imputer(missing_values='NaN',strategy = 'median',axis =0)

#temp = imp.fit_transform(starbucksDF[['Latitude','Longitude']])

#starbucksDF['Latitude'] = temp[:,0]

#starbucksDF['Longitude'] = temp[:,1]
starbucksDF.columns
startbucksDFSortedOnDistance.sort_values(['DistanceDiff'], ascending=[1])
startbucksDFSortedOnDistanceCloseProximity = startbucksDFSortedOnDistance[(startbucksDFSortedOnDistance.DistanceDiff == 0)]
startbucksDFSortedOnDistanceCloseProximity.sort_values(['Latitude','Longitude'], ascending=[1,1])
def get_distance(x, y):

    import math

    NaN=float('nan')

    if (math.isnan(x)):

        return NaN

    else:

        return vincenty((x,y), (-46.41,168.35)).meters

    #.miles



starbucksDF['Distance'] = starbucksDF.apply(lambda row: get_distance(row['Latitude'], row['Longitude']), axis=1)

starbucksDF.sort_values(['Distance'], ascending=[1])
startbucksDFSortedOnDistance=starbucksDF.sort_values(['Distance'], ascending=[1])
import math

NaN=float('nan')

DistanceDiff = []

PrevDistance = []

for row in startbucksDFSortedOnDistance['Distance']:

    # if more than a value,

    if row > 0 and ~(math.isnan(row)):

        # Append a letter grade

        distancediff=row - prev_distance 

        DistanceDiff.append(distancediff)    

        PrevDistance.append(prev_distance)  

    else:

        DistanceDiff.append(NaN)  

        PrevDistance.append(NaN)  

    prev_distance = row

startbucksDFSortedOnDistance['DistanceDiff'] = DistanceDiff

startbucksDFSortedOnDistance['PrevDistance'] = PrevDistance
startbucksDFSortedOnDistance.sort_values(['DistanceDiff'], ascending=[1])
startbucksDFSortedOnDistanceCloseProximity = startbucksDFSortedOnDistance[(startbucksDFSortedOnDistance.DistanceDiff == 0.0)]
startbucksDFSortedOnDistanceCloseProximity.shape
starbucksDF.shape
# 5000+ starbucks stores across world have another starbucks store in close proximity (less than a mile away)
import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as pl
import seaborn as sns
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986

#mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
#asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

#lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
#clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

z = zipfile.ZipFile('../input/train.csv.zip')
df = pd.read_csv(z.open('train.csv'))

z = zipfile.ZipFile('../input/test.csv.zip')
test = pd.read_csv(z.open('test.csv'))




#Treatment of Dates -> Keep only the hour
def hr_func(ts):
    return (float)(ts[11:13])
df['Dates'] = df['Dates'].apply(hr_func)
#Treatment of Hour -> Circular
df['HourCos']=0
df['HourSin']=0

def hourtocos(ts):
    ts=ts*2*math.pi/24
    return math.cos(ts)

def hourtosin(ts):
    ts=ts*2*math.pi/24
    return math.sin(ts)


df['HourCos']=df['Dates'].apply(hourtocos)
df['HourSin']=df['Dates'].apply(hourtosin)



X = df[['X','Y','HourCos','HourSin']]
Y=df[['Category']]
clf=RandomForestClassifier(n_estimators=100,min_samples_split=350)
clf.fit(X,Y)

outcomes = clf.predict_proba()






outcomes<0.01
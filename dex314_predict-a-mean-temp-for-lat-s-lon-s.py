## Going to take a look at the global temps and see if we can make a random
## forest prediction of the mean temperature based on latitude and longitude.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

df_gltbc = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')
df_gltbc = df_gltbc.fillna(0)
## Put together a year column to summarize the data by year.
year = []
for j in df_gltbc['dt']:
    yr = int(j[:4])
    year.append(yr)
df_gltbc['yr'] = year

#%%============================================================
## Might be smart to define a function for this...
Lat = []
Lon = []
## Convert N , S, E, W to qudrant specification.
for row in df_gltbc['Latitude']:
    lenr = len(row)
    if row[(lenr-1):] == 'N':
        Lat.append(float(row[:(lenr-1)]))
    else:
        Lat.append(float(row[:(lenr-1)])*-1)

for row in df_gltbc['Longitude']:
    lenr = len(row)
    if row[(lenr-1):] == 'E':
        Lon.append(float(row[:(lenr-1)]))
    else:
        Lon.append(float(row[:(lenr-1)])*-1)
        
df_gltbc['Lat'] = Lat
df_gltbc['Lon'] = Lon

plt.figure()
plt.scatter(Lat,Lon)
plt.title('Latitude and Longitude Coordinates of Major Cities')
## Pretty cool scatter... can we set a Mercator Projection behind this chart?


#%%============================================================
## Taking a look at temperatures for all cities by year.
byr = df_gltbc.groupby('yr')
byr_avt = byr['AverageTemperature'].agg([np.mean,np.std,np.median,len])

## Temps for each country over the historical course.
byc = df_gltbc.groupby('City')
byc_avt = byc['AverageTemperature'].agg([np.mean,np.std,np.median,len])
df_smz = byc['AverageTemperature'].agg([np.mean])

## Lets make a new data frame summarizing our cities Latitudes, Longitudes, 
## and mean temperatures over the years.
bylat = byc['Lat'].agg([np.mean])
bylon = byc['Lon'].agg([np.mean])

df_smz['lat'] = bylat
df_smz['lon'] = bylon

#%%============================================================
from collections import Counter
city_count = Counter(df_gltbc.City) #confirmed 100 cities
country_count = Counter(df_gltbc.Country) #49 countries
year_count = Counter(df_gltbc.yr) #span of 271 years

## Data merge to get the means with the each represtative city.
city2 = byc_avt.index
byc_avt['City'] = city2
df_mrg = pd.merge(df_gltbc,byc_avt,on='City')

#%%============================================================
## I was thinking to myself can we predict the mean temp from just Lat & Lon?
## After some trial an error were going to go with good ol decision tree regression.
from sklearn import tree

train0 = df_smz

## Lets shuffle the training set and grab the first 40 entries.
nsamp = 40
train_shuf = train0.iloc[np.random.permutation(len(train0))]
train_drop = train_shuf.drop(['mean'],axis=1)
train = train_drop.head(n=nsamp) 
trn_pct = float(len(train))/float(len(train0)) * 100
print('Pecentage of Training Set is ', float(trn_pct))

## Target is the tags.
target = np.asarray(train_shuf['mean'].head(n=nsamp)) #decisiontreereg

clf = tree.DecisionTreeRegressor()
clf.fit(train, target)

#%%===========================================================
## Shuffle the training data randomly for testing the model.

acc = []
for j in range(0,20):
    test_set = train0.iloc[np.random.permutation(len(train0))]
    test1 = test_set.head(n=30) #Take the top 30 after the shuffle.
    target1 = np.asarray(test1['mean'])
    test1 = test1.drop(['mean'],axis=1)
    prd1 = clf.predict(test1)
    acc1 = clf.score(test1,target1) * 100
    acc.append(acc1)

print('Test Accuracies ', acc)

## This is way better. And if you remove the 'head' specification 
## on the validation loop you get the same accuracy
## but different on a per run basis. Maybe we can improve this going further. 

# Lets look at scatter plots of our trainer and our last test we ran...
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig1 = plt.figure()
ax1 = fig1.gca(projection='3d')
z1 = train0["mean"]
x1 = train0["lat"]
y1 = train0["lon"]
ax1.set_title('Training Data')
ax1.scatter(x1, y1, z1, label='Lat/Lon/Mean')
ax1.legend()

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')
z2 = np.reshape(prd1,[len(prd1),1])
x2 = test1["lat"]
y2 = test1["lon"]
ax2.set_title('Last Prediction')
ax2.scatter(x2, y2, z2, label='Lat/Lon/Mean')
ax2.legend()

## It would be really cool if we can set up some kind of contour with the training data and then plot the test data over that to see how it performed from a visual perspective. Im open to any suggestions on how to do that! 


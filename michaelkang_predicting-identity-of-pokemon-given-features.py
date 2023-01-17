# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



from mpl_toolkits.basemap import Basemap

#from matplotlib import animation



from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('../input/300k.csv', low_memory=False)
train[['city','latitude', 'longitude', 'appearedLocalTime']].head(10)
#Used code from Kostya Bahshetsyan's data visualization. Todo: learn how basemap functions.

plt.figure(1, figsize=(20,10))

m1 = Basemap(projection='merc',

             llcrnrlat=-60,

             urcrnrlat=65,

             llcrnrlon=-180,

             urcrnrlon=180,

             lat_ts=0,

             resolution='c')





m1.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes

m1.drawmapboundary(fill_color='#000000')                # black background

m1.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders



# Plot the data

x, y = m1(train.longitude.tolist(),train.latitude.tolist())

m1.scatter(x,y, s=3, c="#1292db", lw=0, alpha=1, zorder=5)

plt.title("Pokemon activity")

plt.show()
NAs = pd.concat([train.isnull().sum()], axis=1)

NAs[NAs.sum(axis=1) > 0]
train = train.drop(['_id', 'cellId_90m', 'cellId_180m', 'cellId_370m', 'cellId_730m', 'cellId_1460m', 'cellId_2920m', 'cellId_5850m'],1)

train = train.drop(['gymIn100m', 'gymIn250m', 'gymIn500m', 'gymIn1000m', 'gymIn2500m', 'gymIn5000m', 'pokestopIn100m', 'pokestopIn250m', 'pokestopIn500m', 'pokestopIn1000m', 'pokestopIn2500m', 'pokestopIn5000m'],1)

train = train.drop(['appearedDayOfWeek'],1)
#Noticed that the appeared Hour/Minute/Day/Month/Year weren't consisted with appearedLocalTime. Removed them all in favor of appearedLocalTime

train = train.drop(['appearedHour', 'appearedMinute', 'appearedDay', 'appearedMonth', 'appearedYear'],1)

#Convert appearedLocalTime string to DateTime

train['appearedLocalTime'] =  pd.to_datetime(train['appearedLocalTime'], format='%Y-%m-%dT%H:%M:%S')        #Note that %y is a 2digit, while %Y is 4digits for the year

#Now reinstate the appeared Hour/Minute/Day/Month/Year, then drop appearedLocalTime

train['appearedHour'] = train['appearedLocalTime'].dt.hour

train['appearedMinute'] = train['appearedLocalTime'].dt.minute

train['appearedDay'] = train['appearedLocalTime'].dt.day

train['appearedMonth'] = train['appearedLocalTime'].dt.month

train['appearedYear'] = train['appearedLocalTime'].dt.year

train = train.drop(['appearedLocalTime'],1)

#Now use 1-of-K encoding using pd.get_dummies()

Hour = pd.get_dummies(train.appearedHour, drop_first=True, prefix='hour')

Minute = pd.get_dummies(train.appearedMinute, drop_first=True, prefix='minute')

Day = pd.get_dummies(train.appearedDay, drop_first=True, prefix='day')

Month = pd.get_dummies(train.appearedMonth, drop_first=True, prefix='month')

Year = pd.get_dummies(train.appearedYear, drop_first=True, prefix='year')

train = train.join(Hour)         #To avoid dummy variable trap

train = train.join(Minute)

train = train.join(Day)

train = train.join(Month)

train = train.join(Year)

#Now we drop the appearedTimeX feature

train = train.drop(['appearedHour', 'appearedMinute', 'appearedDay', 'appearedMonth', 'appearedYear'],1)
#Converting appearedTimeofDay into ordinal

time_mapping = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}

train['appearedTimeOfDay'] = train['appearedTimeOfDay'].map(time_mapping)
#Same for terrainType

Terr = pd.get_dummies(train.terrainType, drop_first=True, prefix='terr')

#train = train.join(Terr)         #To avoid dummy variable trap

#Now we drop the terrain feature

#train = train.drop(['terrainType'],1)
#Get dummies on cities

City = pd.get_dummies(train.city, drop_first=True, prefix='city')

train = train.join(City)         #To avoid dummy variable trap

#Now we drop the city feature

train = train.drop(['city'],1)
#redefining continents such that they correspond to the main 7 continents (no Antartica, yes Indian)

train.continent[train['continent']=='America/Indiana']='America'

train.continent[train['continent']=='America/Kentucky']='America'

train.continent[train['continent']=='Pacific']='Australia'

train.continent[train['continent']=='Atlantic']='Europe'

train.continent[train['continent']=='America/Argentina']='CentralAmerica'

#Then change them to dummies

Continent = pd.get_dummies(train.continent, drop_first=True, prefix='continent')

train = train.join(Continent)         #To avoid dummy variable trap

#Now we drop the continent feature

train = train.drop(['continent'],1)
#Comparing weather columns and choosing to drop weatherIcon. Then use dummies for weather

train['weather'].value_counts()

train['weatherIcon'].value_counts()             #These weather icons are based on time of day as well, making me inclined to not use them.

Weather = pd.get_dummies(train.weather, drop_first=True, prefix='weather')

train = train.join(Weather)         #To avoid dummy variable trap

#Now we drop both weather features

train = train.drop(['weatherIcon', 'weather'],1)
#Want to band windBearing into the 8 cardinal directions. (Probably used azimuth degrees where blowing north is 0 degrees and blowing west is 90 degrees)

#We define North as 0, NW as 1, W as 2, etc...

train.loc[(train['windBearing'] >= 337.5), 'windBearing'] = 0

train.loc[(train['windBearing'] < 22.5), 'windBearing'] = 0

train.loc[(train['windBearing'] >= 22.5) & (train['windBearing'] < 67.5), 'windBearing'] = 1

train.loc[(train['windBearing'] >= 67.5) & (train['windBearing'] < 112.5), 'windBearing'] = 2

train.loc[(train['windBearing'] >= 112.5) & (train['windBearing'] < 157.5), 'windBearing'] = 3

train.loc[(train['windBearing'] >= 157.5) & (train['windBearing'] < 202.5), 'windBearing'] = 4

train.loc[(train['windBearing'] >= 202.5) & (train['windBearing'] < 247.5), 'windBearing'] = 5

train.loc[(train['windBearing'] >= 247.5) & (train['windBearing'] < 292.5), 'windBearing'] = 6

train.loc[(train['windBearing'] >= 292.5) & (train['windBearing'] < 337.5), 'windBearing'] = 7

#Now make them into dummies

WindBearing = pd.get_dummies(train.windBearing, drop_first=True, prefix='windBearing')

train = train.join(WindBearing)         #To avoid dummy variable trap

#Now we drop the wind direction feature

train = train.drop(['windBearing'],1)
#Some quick functions for converting minutes for sunrise/sunset minute standardization

def OnlyPositiveTime(x):

    if x<0:

        return x+1440                   #Where 1440 = minutes per day

    else:

        return x

    

def OnlyNegativeTime(x):

    if x>0:

        return x-1440                   #Where 1440 = minutes per day

    else:

        return x
#Turned Sunrise/set Hour & Minute into dummies. Made sure that minutes since midnight for sunrise/set is positive (no negative minutes)

SunriseHour = pd.get_dummies(train.sunriseHour, drop_first=True, prefix='sunriseHour')

SunriseMinute = pd.get_dummies(train.sunriseMinute, drop_first=True, prefix='sunriseMinute')

SunsetHour = pd.get_dummies(train.sunsetHour, drop_first=True, prefix='sunsetHour')

SunsetMinute = pd.get_dummies(train.sunsetMinute, drop_first=True, prefix='sunsetMinute')

train = train.join(SunriseHour)         #To avoid dummy variable trap

train = train.join(SunriseMinute)

train = train.join(SunsetHour)

train = train.join(SunsetMinute)

#Now we drop the sunrise/set time features

train = train.drop(['sunriseHour', 'sunriseMinute', 'sunsetHour', 'sunsetMinute'],1)

train['sunriseMinutesMidnight'].apply(OnlyPositiveTime)

train['sunsetMinutesMidnight'].apply(OnlyPositiveTime)

#Make sure that each sighting's minutes since sunrise (sunriseMinutesSince) is positive & that sunsetMinutesBefore is negative

train['sunriseMinutesSince'].apply(OnlyPositiveTime)

train['sunsetMinutesBefore'].apply(OnlyNegativeTime)
#Change urban-suburban-urban into numeric values. 0=urban, 1=midurban, 2=suburban, 3=rural

#Dropping suburban and midurban columns, since they dont seem to be accurate. A sighting can't be both urban, suburban, and midurban if they are partitioned bands of population density

#Instead banding to get the urban, suburban, midurban, rural categorization, then changing to ordinal

train = train.drop(['urban', 'suburban', 'midurban', 'rural'],1)

train.loc[train['population_density'] < 200, 'population_density'] = 0

train.loc[(train['population_density'] >= 200) & (train['population_density'] < 400), 'population_density'] = 1

train.loc[(train['population_density'] >= 400) & (train['population_density'] < 800), 'population_density'] = 2

train.loc[train['population_density'] > 800, 'population_density'] = 3

#Just changing the name to show that I processed

train.rename(columns={'population_density' : 'Urbanity'}, inplace = True)
#Changing pokestopDistanceKm from a str to a float

PokestopDistance = pd.to_numeric(train['pokestopDistanceKm'], errors='coerce')

temporary = pd.concat([train, PokestopDistance], axis=1)

#This ends up dropping 39 instances. I'll find out what is causing the NaN's later (Note: errors='coerce' made them NaN's)

train = temporary.dropna()
#Making sure that pokemonID (the first column)) and class (the last column) are the same

row_ids = train[train['class'] != train.pokemonId].index        #This yields an empty set --> identical columns

#So now drop one of them and keep the other (for now) to use as the labels

train.drop(['class'],1)
train_features = train.drop(['pokemonId'],1)

train_labels = train['pokemonId']

X_train, X_test, Y_train, Y_test = train_test_split(train_features, train_labels, train_size = 0.7, random_state = 46)

X_train.shape, Y_train.shape, X_test.shape
model = BernoulliNB()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

acc_1 = round(accuracy_score(Y_test, Y_pred)*100, 2)

acc_1

model = KNeighborsClassifier(n_neighbors = 3)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

acc_3 = round(accuracy_score(Y_test, Y_pred)*100, 2)
model = GaussianNB()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

acc_5 = round(accuracy_score(Y_test, Y_pred)*100, 2)
model = Perceptron()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

acc_6 = round(accuracy_score(Y_test, Y_pred)*100, 2)
model = SGDClassifier()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

acc_8 = round(accuracy_score(Y_test, Y_pred)*100, 2)
model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

acc_9 = round(accuracy_score(Y_test, Y_pred)*100, 2)
models = pd.DataFrame({

    'Model' : ['BernoulliNB', 'KNeighbors', 'Gaussian', 'Perceptron',  'Stochastic Gradient Decent', 'Decision Tree'],

    'Accuracy Score' : [acc_1, acc_3, acc_5, acc_6, acc_8, acc_9]

    })

models.sort_values(by='Accuracy Score', ascending=False)
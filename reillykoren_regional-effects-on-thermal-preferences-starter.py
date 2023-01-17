# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os, random, math, glob

from IPython.display import Image as IM

from IPython.display import clear_output

from matplotlib import pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = [16, 10]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/ashrae-global-thermal-comfort-database-ii/ashrae_db2.01.csv')

#reading in the data
drop_col = ['Publication (Citation)','Data contributor','activity_10','activity_20','activity_30','activity_60','Air temperature (F)','Operative temperature (F)','Ta_h (F)','Ta_m (F)','Ta_l (F)','Radiant temperature (F)','Globe temperature (F)',

    'Tg_h (F)','Tg_m (F)','Tg_l (F)','Tg_h (C)','Tg_m (C)','Tg_l (C)','Air velocity (fpm)','Velocity_h (m/s)','Velocity_m (m/s)','Velocity_l (m/s)','Velocity_h (fpm)','Velocity_m (fpm)','Velocity_l (fpm)','Outdoor monthly air temperature (F)','Database']

data = data.drop(drop_col,axis=1)
tropical_A = []

dry_B = []

temperate_C = []

continental_D = []

polar_E = []



for climate in data['Koppen climate classification'].unique():

    if climate[0] == 'A':

        tropical_A.append(climate)

    elif climate[0] == 'B':

        dry_B.append(climate)

    elif climate[0] == 'C':

        temperate_C.append(climate)

    elif climate[0] == 'D':

        continental_D.append(climate)

    elif climate[0] == 'E':

        polar_E.append(climate)



data.loc[data['Koppen climate classification'].isin(tropical_A), 'Climate'] = 'Tropical'

data.loc[data['Koppen climate classification'].isin(dry_B), 'Climate'] = 'Dry'

data.loc[data['Koppen climate classification'].isin(temperate_C),'Climate'] = 'Temperate'

data.loc[data['Koppen climate classification'].isin(continental_D), 'Climate'] = 'Continental'

data.loc[data['Koppen climate classification'].isin(polar_E),'Climate'] = 'Polar'
data.head(50)
import missingno as msno



msno.matrix(data.select_dtypes(include='number'));
data['Air temperature (C)'].isnull().sum()

#we're missing about a decent chunk of the data
data['Air temperature (C)'] = data.apply(

    lambda column: column['Ta_h (C)'] if np.isnan(column['Air temperature (C)']) else column['Air temperature (C)'],

    axis=1

)



data = data.dropna(how='any', subset=['Air temperature (C)'])

#at this point, we can drop any row that's missing temperature data



msno.matrix(data);
data['Thermal sensation'].isnull().sum()
data['Thermal sensation'].value_counts()
data['Thermal comfort'].value_counts()
data['Thermal comfort'].describe()
#data.loc[data['Thermal sensation'].isnull() & data['Thermal comfort'].astype(float) >= 4.0, 'Thermal sensation'] = 0.0

#data['Thermal sensation'].isnull().sum()
data['Thermal sensation'].describe()

#standard deviation is ~1.25

#if we want to fill these data points, we could potentially use thermal preference data and replace it with a corresponding value

#'warmer' : -1.25

#'colder' : 1.25

#'no change' : 0.0

# some code that may help with this:

# data1['Thermal sensation'] = data1['Thermal sensation'].fillna(data['Thermal preference'].map({'no change':0.0, 'warmer':-1.25,'cooler':1.25})



# Thermal sensation is also missing about 10,000 data points as well, so keep that in mind

# for now, lets drop the missing values

# Let's call this data set "Data Temp"

dataTemp = data.dropna(how='any', subset=['Thermal sensation'])

dataTemp['Thermal sensation'].isnull().sum()
import seaborn as sns



sns.violinplot(x = dataTemp['Thermal sensation'].round(0), y=dataTemp['Air temperature (C)'])
dataTemp = dataTemp[dataTemp['Air temperature (C)'].round(0) >= 10.0]

dataTemp = dataTemp[dataTemp['Air temperature (C)'].round(0) <= 40.0]



dataTemp['Air temperature (C)'].describe()
dataTemp['Country'].value_counts().plot(kind='barh', figsize=(20,6))
uk = dataTemp[dataTemp['Country'] == "UK"]

india = dataTemp[dataTemp['Country'] == "India"]

usa = dataTemp[dataTemp['Country'] == "USA"]

australia = dataTemp[dataTemp['Country'] == "Australia"]

brazil = dataTemp[dataTemp['Country'] == "Brazil"]

china = dataTemp[dataTemp['Country'] == "China"]
def checkCorr(df):

    return df.corr()['Thermal sensation'].sort_values(ascending=False).head(10)



checkCorr(usa)
countries = [uk,india,usa,australia,brazil,china]
c = []

temp = []

sens = []

for country in countries:

    c.append(country["Country"].iloc[0])

    temp.append(country["Air temperature (C)"].mean())

    sens.append(country["Thermal sensation"].mean())



meanDF=pd.DataFrame()

meanDF["Country"] = c

meanDF["Avg Room Temp"] = temp

meanDF["Avg Thermal Sensation"] = sens

meanDF.head(6)



# meanData = pd.DataFrame(np.array([c, temp, sens]),columns=['Country', 'Mean Room Temperature (C)', 'Mean Thermal Sensation'])

med_0to1 = []

med_1to0 = []

med_gt1 = []

med_lt1 = []

med_0 = []



for country in countries:

    med_0to1.append(country[(country["Thermal sensation"] <= 1.0) & (country["Thermal sensation"] > 0.0)]["Air temperature (C)"].median())

    med_1to0.append(country[(country["Thermal sensation"] >= -1.0) & (country["Thermal sensation"] < 0.0)]["Air temperature (C)"].median())

    med_gt1.append(country[country["Thermal sensation"] > 1.0]["Air temperature (C)"].median())

    med_lt1.append(country[country["Thermal sensation"] < -1.0]["Air temperature (C)"].median())

    med_0.append(country[country["Thermal sensation"] == 1.0]["Air temperature (C)"].median())

    

pct_0to1 = []

pct_1to0 = []

pct_gt1 = []

pct_lt1 = []

pct_0 = []



for country in countries:

    pct_0to1.append(country[(country["Thermal sensation"] <= 1.0) & (country["Thermal sensation"] > 0.0)].shape[0]/country.shape[0] * 100)

    pct_1to0.append(country[(country["Thermal sensation"] >= -1.0) & (country["Thermal sensation"] < 0.0)].shape[0]/country.shape[0] * 100)

    pct_gt1.append(country[country["Thermal sensation"] > 1.0]["Air temperature (C)"].shape[0]/country.shape[0] * 100)

    pct_lt1.append(country[country["Thermal sensation"] < -1.0]["Air temperature (C)"].shape[0]/country.shape[0] * 100)

    pct_0.append(country[country["Thermal sensation"] == 0.0]["Air temperature (C)"].shape[0]/country.shape[0] * 100)

    

meanDF["Median Temp 0 to 1.0"] = med_0to1

meanDF["Median Temp -1.0 to 0"] = med_1to0

meanDF["Median Temp > 1.0"] = med_gt1

meanDF["Median Temp < -1.0"] = med_lt1

meanDF["Median Temp 0.0"] = med_0



meanDF["Pct Temp 0 to 1.0"] = pct_0to1

meanDF["Pct Temp -1.0 to 0"] = pct_1to0

meanDF["Pct Temp > 1.0"] = pct_gt1

meanDF["Pct Temp < 1.0"] = pct_lt1

meanDF["Pct Temp 0.0"] = pct_0



meanDF.head(6)
df = pd.DataFrame({'Avg Room Temp': temp,'% Positive': pct_0}, index=c)

axes = df.plot.bar(rot=0, subplots=True)

axes[1].legend(loc=0)
country = uk #change as needed

in_cols = country['PMV']

in_cols







from sklearn.metrics import mean_squared_error

from math import sqrt

country1 = country.dropna(subset=['PMV','Thermal sensation'])

#np.any(np.isnan(country['Thermal sensation']))

print("RMSE:", sqrt(mean_squared_error(country1['Thermal sensation'][-1000:], country1['PMV'][-1000:])))

med_0to1 = country[(country["Thermal sensation"] <= 1.0) & (country["Thermal sensation"] > 0.0)]["Air temperature (C)"].median()

med_1to0 = country[(country["Thermal sensation"] >= -1.0) & (country["Thermal sensation"] < 0.0)]["Air temperature (C)"].median()

med_gt1 = country[(country["Thermal sensation"] > 1.0)]["Air temperature (C)"].median()

med_lt1 = country[(country["Thermal sensation"] < -1.0)]["Air temperature (C)"].median()

med_0 = country[(country["Thermal sensation"] == 0.0)]["Air temperature (C)"].median()



pct_0to1 = country[(country["Thermal sensation"] <= 1.0) & (country["Thermal sensation"] > 0.0)].shape[0]/country.shape[0]

pct_1to0 = country[(country["Thermal sensation"] >= -1.0) & (country["Thermal sensation"] < 0.0)].shape[0]/country.shape[0]

pct_0 = country[country["Thermal sensation"] == 0.0]["Air temperature (C)"].shape[0]/country.shape[0]

leftLine = med_0 - ((pct_1to0 / (pct_0 + pct_1to0)) * (med_0 - med_1to0))

rightLine = med_0 + ((pct_0to1 / (pct_0 + pct_0to1)) * (med_0to1 - med_0))





prediction = []

for temp in country['Air temperature (C)']:

    if(temp > med_gt1):

        prediction.append(2.0)

    elif(temp > rightLine):

        prediction.append(1.0)

    elif(temp > leftLine):

        prediction.append(0.0)

    elif(temp > med_lt1):

        prediction.append(-1.0)

    else:

        prediction.append(-2.0)



country['Prediction'] = prediction

print("RMSE:", sqrt(mean_squared_error(country['Thermal sensation'][-1000:], country['Prediction'][-1000:])))



        
country = usa

country['City'].value_counts().plot(kind='barh', figsize=(20,6))


honolulu = country[country['City'] == "Honolulu"]

philly = country[country['City'] == "Philadelphia"]

sanfran = country[country['City'] == "San Francisco"]

grandrap = country[country['City'] == "Grand Rapids"]



cities = [honolulu,philly,sanfran,grandrap]

c2 = []

temp2 = []

sens2 = []

for city in cities:

    c2.append(city["City"].iloc[0])

    temp2.append(city["Air temperature (C)"].mean())

    sens2.append(city["Thermal sensation"].mean())



meanDF=pd.DataFrame()

meanDF["City"] = c2

meanDF["Avg Room Temp"] = temp2

meanDF["Avg Thermal Sensation"] = sens2

meanDF.head(4)
med_0to1 = []

med_1to0 = []

med_gt1 = []

med_lt1 = []

med_0 = []



for city in cities:

    med_0to1.append(city[(city["Thermal sensation"] <= 1.0) & (city["Thermal sensation"] > 0.0)]["Air temperature (C)"].median())

    med_1to0.append(city[(city["Thermal sensation"] >= -1.0) & (city["Thermal sensation"] < 0.0)]["Air temperature (C)"].median())

    med_gt1.append(city[city["Thermal sensation"] > 1.0]["Air temperature (C)"].median())

    med_lt1.append(city[city["Thermal sensation"] < -1.0]["Air temperature (C)"].median())

    med_0.append(city[city["Thermal sensation"] == 1.0]["Air temperature (C)"].median())

    

pct_0to1 = []

pct_1to0 = []

pct_gt1 = []

pct_lt1 = []

pct_0 = []



for city in cities:

    pct_0to1.append(city[(city["Thermal sensation"] <= 1.0) & (city["Thermal sensation"] > 0.0)].shape[0]/city.shape[0] * 100)

    pct_1to0.append(city[(city["Thermal sensation"] >= -1.0) & (city["Thermal sensation"] < 0.0)].shape[0]/city.shape[0] * 100)

    pct_gt1.append(city[city["Thermal sensation"] > 1.0]["Air temperature (C)"].shape[0]/city.shape[0] * 100)

    pct_lt1.append(city[city["Thermal sensation"] < -1.0]["Air temperature (C)"].shape[0]/city.shape[0] * 100)

    pct_0.append(city[city["Thermal sensation"] == 0.0]["Air temperature (C)"].shape[0]/city.shape[0] * 100)

    

meanDF["Median Temp 0 to 1.0"] = med_0to1

meanDF["Median Temp -1.0 to 0"] = med_1to0

meanDF["Median Temp > 1.0"] = med_gt1

meanDF["Median Temp < -1.0"] = med_lt1

meanDF["Median Temp 0.0"] = med_0



meanDF["Pct Temp 0 to 1.0"] = pct_0to1

meanDF["Pct Temp -1.0 to 0"] = pct_1to0

meanDF["Pct Temp > 1.0"] = pct_gt1

meanDF["Pct Temp < 1.0"] = pct_lt1

meanDF["Pct Temp 0.0"] = pct_0



meanDF.head(4)
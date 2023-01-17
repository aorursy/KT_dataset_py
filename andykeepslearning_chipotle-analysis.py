# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

from shapely.geometry import Point

from shapely import wkt

import folium

from folium import Choropleth



import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

pd.plotting.register_matplotlib_converters()

%matplotlib inline

import seaborn as sns



import sklearn

from sklearn import linear_model



import statsmodels.regression.linear_model as sm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
chipotle = gpd.read_file("../input/chipotle-locations/chipotle_stores.csv")

geometry = [Point(xy) for xy in zip(chipotle['longitude'].astype('float'),chipotle['latitude'].astype('float'))]

chipotle['geometry'] = geometry

chipotle.head()



us_states = gpd.read_file("../input/chipotle-locations/us-states.json")

us_counties = gpd.read_file('../input/enrichednytimescovid19/us_county_pop_and_shps.csv')

us_counties = us_counties.loc[us_counties['county_geom'] != 'None']



census2017 = gpd.read_file('../input/us-census-demographic-data/acs2017_county_data.csv')



covid2019 = gpd.read_file('../input/covid19-us-county-jhu-data-demographics/us_county.csv')
us_counties['county_geom'] = us_counties['county_geom'].apply(wkt.loads)

us_counties = gpd.GeoDataFrame(us_counties, geometry='county_geom')

census2017
chipotle_enhanced = gpd.sjoin(chipotle,us_counties)

chipotle_enhanced.head()
#calculate population density as people / km**2

us_counties['popdensity'] = us_counties['county_pop_2019_est'].astype(float) / (us_counties['county_geom'].area * 10000)



#create a common key for reading in calculated density

chipotle_enhanced['statecounty'] = chipotle_enhanced['state_right'] + chipotle_enhanced['county']

us_counties['statecounty'] = us_counties['state'] + us_counties['county']



#read in calculated density

chipotle_enhanced = pd.merge(chipotle_enhanced, us_counties[['statecounty','popdensity']], on='statecounty',how='left')



#create a common key in the us census data

census2017['statecounty'] = census2017['State'] + census2017['County']

census2017['statecounty'] = census2017['statecounty'].replace(' County','', regex=True)



#create a common key in the us county covid data

covid2019['statecounty'] = covid2019['state'] + covid2019['county']

covid2019['statecounty'] = covid2019['statecounty'].replace(' County','',regex=True)
#create a pivot table giving the number of stores by county

chipotle_counts = pd.pivot_table(chipotle_enhanced, values=['address'], index=['statecounty'], aggfunc=lambda x: len(x.unique()))



#trim some columns from the census data

#dropping TotalPop because we have a 2019 estimate, which should be better than a 2017 estimate

census2017_to_merge = census2017.drop(['State','County','CountyId','geometry','TotalPop'], axis=1)



#read in other potentially useful fields for a regression

chipotle_counts = pd.merge(chipotle_counts, us_counties[['statecounty','popdensity','county_pop_2019_est']], on='statecounty',how='left')

chipotle_counts = pd.merge(chipotle_counts, census2017_to_merge, on='statecounty',how='left')

chipotle_counts = pd.merge(chipotle_counts, covid2019[['median_age','statecounty']],on='statecounty',how='left')



# #rename the "address" column to "count" because it annoys me

# chipotle_counts.rename(columns={'address':'count'})



#There are some missing values in a few fields.  Filling them in with the next entry, or failing that, zero

chipotle_counts = chipotle_counts.fillna(method='bfill',axis=0).fillna(0)



#convert values to floats so we don't have to keep doing it

#chipotle_counts = chipotle_counts.loc[:, chipotle_counts.columns != 'statecounty'].astype('float64')

chipotle_counts.loc[:, chipotle_counts.columns != 'statecounty'] = chipotle_counts.loc[:, chipotle_counts.columns != 'statecounty'].apply(pd.to_numeric)



#display data to make sure things look about right

chipotle_counts
X = chipotle_counts.drop(['address','statecounty'],axis=1)

y = chipotle_counts.address



lm = linear_model.LinearRegression()

model=lm.fit(X,y)

lm.score(X,y)
mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())
X = chipotle_counts.drop(['address','statecounty','Asian','IncomeErr','PrivateWork','PublicWork','SelfEmployed'],axis=1)



mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())
X = chipotle_counts.drop(['address','statecounty','Asian','IncomeErr','PrivateWork','PublicWork','SelfEmployed','popdensity','Native','FamilyWork'],axis=1)



mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())
X = chipotle_counts.drop(['address','statecounty','Asian','IncomeErr','PrivateWork','PublicWork','SelfEmployed','popdensity','Native','FamilyWork','White','Black','Poverty','ChildPoverty','median_age'],axis=1)



mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())
X = chipotle_counts.drop(['address','statecounty','Asian','IncomeErr','PrivateWork','PublicWork','SelfEmployed','popdensity','Native','FamilyWork','White','Black','Poverty','ChildPoverty','median_age','Men','Hispanic'],axis=1)



mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())
X = chipotle_counts.drop(['address','statecounty','Asian','IncomeErr','PrivateWork','PublicWork','SelfEmployed','popdensity','Native','FamilyWork','White','Black','Poverty','ChildPoverty','median_age','Men','Hispanic','Office','Construction','Production','Drive','Carpool','Transit','WorkAtHome','MeanCommute','Unemployment'],axis=1)



mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())
X = chipotle_counts.drop(['address','statecounty','Asian','IncomeErr','PrivateWork','PublicWork','SelfEmployed','popdensity','Native','FamilyWork','White','Black','Poverty','ChildPoverty','median_age','Men','Hispanic','Office','Construction','Production','Drive','Carpool','Transit','WorkAtHome','MeanCommute','Unemployment','Income','IncomePerCap','Professional','Service','Walk','OtherTransp'],axis=1)



mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())
X = chipotle_counts.drop(['address','statecounty','Asian','IncomeErr','PrivateWork','PublicWork','SelfEmployed','popdensity','Native','FamilyWork','White','Black','Poverty','ChildPoverty','median_age','Men','Hispanic','Office','Construction','Production','Drive','Carpool','Transit','WorkAtHome','MeanCommute','Unemployment','Income','IncomePerCap','Professional','Service','Walk','OtherTransp','VotingAgeCitizen','IncomePerCapErr'],axis=1)



mod = sm.OLS(y, X)

res = mod.fit()

print(res.summary())
#retrain model

X = chipotle_counts.loc[:,['Women','Pacific','Employed','county_pop_2019_est']]

y = chipotle_counts.address

lm = linear_model.LinearRegression()

model=lm.fit(X,y)



#develop new predictions with all counties

census_xpred = census2017.drop(['geometry','CountyId','State','County','TotalPop','Asian','IncomeErr','PrivateWork','PublicWork','SelfEmployed','Native','FamilyWork','White','Black','Poverty','ChildPoverty','Men','Hispanic','Office','Construction','Production','Drive','Carpool','Transit','WorkAtHome','MeanCommute','Unemployment','Income','IncomePerCap','Professional','Service','Walk','OtherTransp','VotingAgeCitizen','IncomePerCapErr'],axis=1)

xpred = pd.merge(census_xpred,us_counties[['statecounty','county_pop_2019_est']], on='statecounty',how='left')

xpred = xpred.dropna()

census2017_with_pred = xpred

xpred = xpred.loc[:, xpred.columns != 'statecounty'].astype('float64')

census2017_with_pred['ypred'] = model.predict(xpred)

census2017_with_pred
#Merging the prediction and actual counts with county geometry

model_data = pd.merge(us_counties,census2017_with_pred, on='statecounty',how='left')

model_data['statecounty'] = model_data['state'] + model_data['county']

model_data = pd.merge(model_data, chipotle_counts[['address','statecounty']],on='statecounty',how='left')

model_data.address.fillna(0, inplace=True)

model_data.ypred.fillna(0, inplace=True)



#Normally we'd subtract "actual" minus "expected" to determine residuals.  In this case, 

#it would be better to have "expected" minus "actual" to show how many restaurants should be built.



model_data['residuals'] = model_data['ypred'] - model_data['address']

model_data.crs = "epsg:4326"

model_data
print('Maximum residual = ' + str(model_data.residuals.max()))

print('Minimum residual = ' + str(model_data.residuals.min()))
geo_data = model_data[['statecounty','county_geom']].set_index('statecounty')

residual_data = model_data[['statecounty','residuals']].set_index('statecounty')



m_1 = folium.Map(location=[40,-100], tiles='openstreetmap', zoom_start=4)



Choropleth(geo_data=geo_data.__geo_interface__, data=residual_data['residuals'], fill_color='RdYlGn', key_on='feature.id', legend_name='Chipotles to build per county', threshold_scale=[-50,-1,1,5,10,12]).add_to(m_1)



m_1
model_data.sort_values('residuals',ascending=False).head(10)
model_data.sort_values('residuals',ascending=True).head(10)
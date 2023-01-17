# First lets import the libraries
import pandas as pd
import datetime as dt
import re
import numpy as np
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Setting pandas display options and column names for the dataset
pd.set_option('display.max_columns', 500)
cols = ['datetime','city','state','country','shape','seconds',
              'minutes','comments','date added','lat','long']

# loading the uncleaned dataset and assign columns names
df = pd.read_csv('../input/ufo-sightings/scrubbed.csv', sep=',',names =cols)
df = df.drop(0, axis=0)
df.head(3)
df.info()
df.isna().sum()
#Using Fillna To Impute 'unspecified' in place of NaN values, will deal with numeric cols classed as 'object' later
df = df.fillna('unspecified')
df.shape
# Extracting Date and Time from datetime Col into sighting_date(Datetime64) and time(int) Columns
df['sighting_date'] = df['datetime'].str.findall(r'\d{1,2}.\d{1,2}.\d{1,4}').str.join('')
df['time'] = df['datetime'].str.findall(r'\s\d{1,2}.\d{1,2}').str.join('').str.replace(':', '')
df['sighting_date'] = pd.to_datetime(df['sighting_date'])
df['time'] = df['time'].astype(int)
df.shape
# Converting Lat Long Cols To Floats 
df[['lat', 'long']] = df[['lat', 'long']].astype(float)
# Removing Row Typos And The String Fillna From Earlier
df['seconds'] = df['seconds'].str.replace('8/16/2002 01:30', '0.0')
df['seconds'] = df['seconds'].str.replace('`','')
df['seconds'] = df['seconds'].str.replace('unspecified', '0.0')
# Updating All 0 Second Sightings to Median(120 secs) 
df['seconds'] = df['seconds'].str.replace('0.0', '120')
df['seconds'] = df['seconds'].str.replace(r'^0', '120', regex=True)

# Converting Seconds To Float
df['seconds'] = df['seconds'].astype(float)

# Updating Minutes Column With Seconds Coversion (Easier To Interperet/acceptable foprmat for modelling)
df['minutes'] = df['seconds'] / 60
# Check the variance of the minutes column
print('Minutes Variance =',df[ 'minutes'].var())

# Log normalize the minutes column to reduce Variance
df["minutes_log"] = np.log(df['minutes'])

# Print out the variance of just the seconds_log column
print('Log Transformed Minutes Variance =',df['minutes_log'].var())
# Maybe We Can Drop The Outliers
plt.hist(df['minutes_log'], bins=10)
plt.xticks(np.arange(-5, 17.5, 2.5))
plt.show()
print('Max Minutes Log:', df['minutes_log'].max() )
# Extracting month and month name from the month column
df["month"] = df["sighting_date"].apply(lambda x: x.month)
df['month_name'] = df['month'].apply(lambda x: calendar.month_name[x])

# Extracting the year from the date column
df["year"] = df["sighting_date"].apply(lambda x: x.year)

# All three columns
print(df[['sighting_date', 'month', 'month_name', 'year']].head(3))
print(df.shape)
# Missing A Lot Of Countries Filled With 'Unspecified'
df['country'].value_counts()
# Less Missing For States 
df['state'].value_counts()
# Setting Region Boundries With Google Maps Coordinates. 
# These Are Broad Continent/Regions and will be added to the 'Countries' Column

conditions = [ 
#aus
(df['lat'] > -48.0) & (df['lat'] <-7.0) & (df['long'] > -171.0) & (df['lat'] < 104.0)
    & (df['lat'] != -0) & (df['long'] != -0), 
#africa                                                                                        
(df['lat'] > -37.0) & (df['lat'] <37.0) & (df['long'] > -26.0) & (df['lat'] < 48.0)
    & (df['lat'] != -0) & (df['long'] != -0),
#mideast
(df['lat'] >10.0) & (df['lat'] <47.0) & (df['long'] > 35.0) & (df['lat'] < 90.0)
    & (df['lat'] != 0) & (df['long'] != 0),
#latam
(df['lat'] >-59.0) & (df['lat'] <30.0) & (df['long'] < -29.0) & (df['lat'] >-126.0)
    & (df['lat'] != -0) & (df['long'] != -0)]


                                                                                       

categories = ['aus-nz', 'africa', 'mideast', 'latam']

df['added_countries'] = np.select(conditions, categories)
df['added_countries'] = df['added_countries'].str.replace('0','')
df['country'] = df['country'] + df['added_countries']
df['added_countries'].value_counts()
df = df.drop('added_countries', axis=1)
# Mapping new regions and combining with existing regions
df['country'] = df['country'].map({'unspecifiedafrica':'africa', 'unspecifiedlatam ': 'latam',
                                  'unspecifiedaus-nz':'aus-nz', 'auaus-nz': 'aus-nz', "au":'aus-nz',
                                  'unspecifiedmideast':'mideast', 'uslatam': 'latam', 'gb':'uk-irl',
                                  'ca':'ca', 'de':'de', 'us':'us', 'unspecified':'unspecified'})

df['country'] = df['country'].fillna('unspecified')
df['country'].value_counts()
# Label Encoding The Country/Regions
le = LabelEncoder()

x = le.fit_transform(df['country'])

df['country_enc'] = x
df['country_enc'].value_counts()
df['state']
# Label Encoding The States
le = LabelEncoder()

y = le.fit_transform(df['state'])

df['state_enc'] = y
# Extracting Cities From A String Copied That I Copied From Google
with open('../input/country-list/country_list.txt') as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]
lines = [x.lower() for x in lines]
lines[:2]
# Extracting The List and Filling NaNs' with 'Unspecified'
df['added_region'] = df['city'].str.findall(f'({"|".join(lines)})')
df['added_region'] = df['city'].str.extract(f'({"|".join(lines)})')
df['added_region'] = df['added_region'].fillna('unspecified')
df['added_region'].value_counts()
# Label Encoding The Added Cities
le = LabelEncoder()

x = le.fit_transform(df['added_region'])

df['city_enc'] = x
# Creating Time Of Day Column
df.loc[(df['time'] >=600) & (df['time'] <= 1200), 'sighting_time'] = 'morning'+'\n'+'(6am-12pm)'
df.loc[(df['time'] >=1200) & (df['time'] <=  1600), 'sighting_time'] = 'afternoon'+'\n'+'(12pm-4pm)'
df.loc[(df['time'] >=1600) & (df['time'] <=  2100), 'sighting_time'] = 'evening'+'\n'+'(4pm-9pm)'
df.loc[(df['time'] >=2100) & (df['time'] <=  2400), 'sighting_time'] = 'nighttime'+'\n'+'(9pm-12am)'
df.loc[(df['time'] >=0) & (df['time'] <=  600), 'sighting_time'] = 'latenight'+'\n'+'(12am-6am)'
# Encoding sighting_time 
def sighting(val):
    if val == 'morning'+'\n'+'(6am-12pm)':
        return 1
    elif val == 'afternoon'+'\n'+'(12pm-4pm)':
        return 2
    elif val =='evening'+'\n'+'(4pm-9pm)':
        return 3
    elif val== 'nighttime'+'\n'+'(9pm-12am)':
        return 4
    else:
        return 5

df['sighting_time_enc'] = df['sighting_time'].apply(sighting)
# Unique shape sighting count
print(df['shape'].value_counts())
# Number Of Shapes
print('\n'+'Number Of Shapes =', len(df['shape'].unique()))
# OHE the ufo shapes
shape_set = pd.get_dummies(df['shape'])

# Concatenating OHE Values to the df 
df = pd.concat([df, shape_set], axis=1)
df.shape
# Smoother More Legible Plotting Style
sns.set(style="white", context="talk")
# Plotting Sighting Over Time (Since 1940)
plt.figure(figsize=(20,10))

sns.lineplot(data=df['year'].value_counts())

plt.xlim(1940, 2018)
plt.xticks(range(1940,2020, 5), rotation=0, size=14)
plt.yticks(range(0,10000, 1000), rotation=0, size=14)

plt.xlabel('Year', fontweight='semibold')
plt.ylabel('Sighting Count', fontweight='semibold')
plt.title('Number of Sightings Per Year', fontweight='semibold')
plt.show()
# Number Of Sightings By Month
plt.figure(figsize=(20,10))
sns.countplot(data=df.sort_values(by='month'),x='month_name')
plt.xlabel('Month Of Sighting', fontweight='semibold')
plt.ylabel('Sighting Count', fontweight='semibold')
plt.title('Number Of Sightings By Month', fontweight='semibold')
plt.show()
# Number Of Sightings By Time Of Day
plt.figure(figsize=(20,10))
sns.countplot(data=df,x='sighting_time', 
              order=df['sighting_time'].value_counts().index)
plt.xlabel('Time Of Day (Sighting)', fontweight='semibold')
plt.ylabel('Sighting Count', fontweight='semibold')
plt.title('Number Of Sightings By Time Of Day', fontweight='semibold')
plt.show()
# Make The Upcoming Labels Easier To Read, Some I will Set To Upper After Grouping
df['state'] = df['state'].apply(lambda x: x.upper())
df['city'] = df['city'].apply(lambda x: x.upper())
df['country'] = df['country'].apply(lambda x: x.upper())
# Sightings By Country/Region
plt.figure(figsize=(20,10))
sns.countplot(data=df, x='country')
plt.xlabel('Worldwide Region', fontweight='semibold')
plt.ylabel('Sighting Count', fontweight='semibold')
plt.title('Sightings By Country/Region', fontweight='semibold')
plt.show()
# Group By States With Over 1000 Sighting AKA Hotspots
by_state = (df
    .groupby('state')
    .filter(lambda x: len(x) > 1000))
# Sighting By State Hotspots
plt.figure(figsize=(20,10))
sns.countplot(x = 'state',
              data = by_state,
              order = by_state['state'].value_counts().index)

plt.xticks(size=16, rotation=90)
plt.yticks(range(0,12000,1000),size=16, rotation=0)
plt.xlabel('State', fontweight='semibold')
plt.ylabel('Sighting Count', fontweight='semibold')
plt.title('US Sighting Hotspots', fontweight='semibold')
plt.show()
# Grouping Cities With Over 200 Sighting
by_city = (df
    .groupby('city')
    .filter(lambda x: len(x) > 200))
# Sighting By Global City Hotspot
plt.figure(figsize=(25,10))
sns.countplot(x = 'city',
              data = by_city,
              order = by_city['city'].value_counts().index)
plt.xlabel('City', fontweight='semibold')
plt.ylabel('Sighting Count', fontweight='semibold')
plt.title('Global City Hotspots', fontweight='semibold')
plt.xticks(rotation=90)
plt.show()
# Groupby Count Transforming Data To Heatmap Compatable Count Of Shapes Per Cities
shape_by_city = (df
    .groupby('city')
    .filter(lambda x: len(x) > 100) 
    .groupby(['shape', 'city'])
    .size()
    .unstack())

# Make The Shapes More Legible
shape_by_city.index = shape_by_city.index.str.upper()
# Shape By City Hotspot
plt.figure(figsize=(30,8))
heat = sns.heatmap(
        shape_by_city,
        square=True,
        cbar_kws= {'fraction': 0.01},
        cmap='Oranges',
        linewidth=1
)
heat.set_xticklabels(heat.get_xticklabels(), fontsize=14, rotation=90,horizontalalignment='center')
heat.set_yticklabels(heat.get_yticklabels(), fontsize=14, rotation=0)
plt.xlabel('Global City', fontweight='semibold')
plt.ylabel('Sighting Count', fontweight='semibold')
plt.title('Sighting Count Per Shape', fontweight='semibold')
plt.show()
shape_by_time = (df
    .groupby('shape')
    .filter(lambda x: len(x) > 1) # Filters Out European Sightings and Lower Activity US Cities
    .groupby(['sighting_time', 'shape'])
    .size()
    .unstack())
shape_by_time.index = shape_by_time.index.str.upper()
shape_by_time.columns = shape_by_time.columns.str.upper()
# Shape By Time Of Day 
plt.figure(figsize=(20,10))
heat = sns.heatmap(
        shape_by_time,
        square=True,
        cbar_kws= {'fraction': 0.01},
        cmap='Oranges',
        linewidth=1
)
heat.set_xticklabels(heat.get_xticklabels(), fontsize=14, rotation=45,horizontalalignment='right')
heat.set_yticklabels(heat.get_yticklabels(), fontsize=14, rotation=0,horizontalalignment='right')
plt.xlabel('Shape', fontweight='semibold')
plt.ylabel('Time Of Day', fontweight='semibold')
plt.title('UFO Shape By Time Of Day', fontweight='semibold')
plt.show()
shape_by_year = (df
    .groupby('shape')
    .filter(lambda x: len(x) > 1) # Filters Out European Sightings and Lower Activity US Cities
    .groupby(['shape', 'year'])
    .size()
    .unstack())
shape_by_year.index = shape_by_year.index.str.upper()
# Shape By Year 
plt.figure(figsize=(20,10))
heat = sns.heatmap(
        shape_by_year,
        square=True,
        cbar_kws= {'fraction': 0.01},
        cmap='Oranges',
        linewidth=1
)
heat.set_xticklabels(heat.get_xticklabels(), fontsize=14, rotation=45,horizontalalignment='right')
heat.set_yticklabels(heat.get_yticklabels(), fontsize=12, rotation=0)
plt.xlabel('Year', fontweight='semibold')
plt.ylabel('UFO Shape', fontweight='semibold')
plt.title('UFO Shape By Year', fontweight='semibold')
plt.show()
# Global Map Of UFO Sightings Using Lat Long
plt.figure(figsize=(30,15))
m = Basemap(projection='mill',
           llcrnrlat = -50,
           urcrnrlat = 90,
           llcrnrlon = -180,
           urcrnrlon = 180,
           resolution = 'c')

m.drawstates()
m.drawcountries()
m.drawcoastlines()

lat, long = df['lat'].tolist(), df['long'].tolist()

t = np.arange(80332)

m.scatter(long, lat, marker = 'o', c=t, cmap='YlOrRd', s=1, zorder=10, latlon=True)
m.fillcontinents(color='g', alpha =0.3)

plt.title("Global UFO Sightings", fontsize=26, fontweight='semibold')

plt.show()
df.head()
df.dtypes
# Dropping Columns That Have Converted/Encoded Copies
no_nan_df = df.drop(['datetime', 'seconds', 'state', 'city', 'minutes','country', 'shape','comments',
              'date added','month_name', 'added_region','sighting_time','sighting_date'],axis=1)
# Check For High Variance 
no_nan_df.var()
minutes_agg = df['minutes_log'].agg([min, np.median, np.mean, max, np.std])
minutes_agg
# Creating Numeric X, y Vairables
X, y = no_nan_df.drop(
    ['minutes_log', 'lat', 'long'],axis=1),no_nan_df['minutes_log']


# Split the X and y sets using train_test_split, setting stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=123)

print(X_train.shape)
print(X_test.shape)
# Basic Tuned XGBRegressor (No SearchCV Performed)
clf_r = xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.01, learning_rate = 1,
                max_depth = 50, alpha = 0.1, n_estimators = 100)


clf_r.fit(X_train,y_train)

preds = clf_r.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Test Set RMSE: %f" % (rmse))
# Cross Validating The Results
data_dmatrix = xgb.DMatrix(data=X,label=y)

params = {"objective":'reg:squarederror','colsample_bytree': 0.9,'learning_rate': 0.7,
                'max_depth': 5, 'alpha': 0.1}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()
no_nan_df.head(2)
# Creating Numeric X, y Vairables
X, y = no_nan_df.drop(
    ['country_enc','lat','long'],
    axis=1),no_nan_df['country_enc']

# Split the X and y sets using train_test_split, setting stratify=y
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=123)

print(X_train.shape)
print(X_test.shape)
# Tuned XGboost Classifier (Tuned With Random Search Below)
clf_c = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.15, max_delta_step=0, max_depth=4,
              min_child_weight=1, n_estimators=50, n_jobs=0, num_parallel_tree=2,
              objective='multi:softprob', random_state=123, reg_alpha=0.1,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=1) 


clf_c.fit(X_train, y_train)

# Test and Training Set Acurracy Score
accuracy_train = accuracy_score(y_train, clf_c.predict(X_train))
accuracy_test = accuracy_score(y_test, clf_c.predict(X_test))
print("{0:.1%} accuracy on test set vs. {1:.1%} on training set".format(accuracy_test, accuracy_train))
# Further Assessing Accuracy Across Classes
y_pred = clf_c.predict(X_test)
print(classification_report(y_test, y_pred))
kfold = KFold(n_splits=10)
results = cross_val_score(clf_c, X, y, cv=kfold)
print("Mean Accuracy: %.2f%%, Standard Deviation (%.2f%%)" % (results.mean()*100, results.std()*100))
# # # Running A GridSearchCV To Find More Optimal Parameters
# xgb_param_grid = {
#     'learning_rate': np.arange(0.05, 1, 0.05),
#     'max_depth': np.arange(5, 12, 2),
#     'n_estimators': np.arange(0, 100, 50),
#     'gamma': np.arange(0, 1, 0.1),
#     'colsample_bylevel': np.arange(0, 1, 0.1)
# }
    
# rndm_xgb = RandomizedSearchCV(estimator=clf_c, param_distributions=xgb_param_grid, verbose=2, 
#                              scoring='precision_weighted', cv=3, n_jobs=3)

# rndm_xgb.fit(X, y)

# # Computing Param metrics
# print('Best precision_weighted: %.2f'% rndm_xgb.best_score_)
# print('Best Params:',rndm_xgb.best_estimator_)
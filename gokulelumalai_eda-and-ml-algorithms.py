# let's import necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



%config Inlinebackground.figureFormat='retina'

sns.set(font_scale=1.5)

pd.options.display.max_rows = 200

pd.options.display.max_columns = 200
# let's load the datasets

# cab data

cab_data = pd.read_csv(r'cab_rides.txt', encoding='utf-16')

# weather data

weather_data = pd.read_csv(r'weather.txt', encoding='utf-16')
cab_data.head(3)
weather_data.head(3)
cab_data.info() # basic descr.
# let's impute the unix epoch time to standard date time format

cab_data['time_stamp'] = pd.to_datetime(cab_data['time_stamp'], unit='ms')

cab_data['date'] = cab_data['time_stamp'].dt.date  # extract date

cab_data['hour'] = cab_data['time_stamp'].dt.hour  # extract hour



cab_data.drop('time_stamp', axis=1, inplace=True)  # drop time_stamp feature



# before doing EDA, let's split the dataset into Uber and Lyft

uber = cab_data[cab_data['cab_type']=='Uber']

lyft = cab_data[cab_data['cab_type']=='Lyft']



cab_data.head(3)
overall = cab_data['distance'].describe() # measure of central tendency

overall
lyft_distance = lyft['distance'].describe()

uber_distance = uber['distance'].describe()
df = pd.DataFrame({'Overall': overall.values,

                  'Lyft': lyft_distance.values,

                  'Uber': uber_distance.values}, index= ['Count', 'Mean', 'Std. Dev.', 'Min', '25%', '50%', '75%', 'Max'])

df
# df.to_csv(r'C:\Users\gokul\Downloads\distance_metrics.csv')
def calculate_mop(**kwargs):

    """ function to calculate and display the measures of dispersion."""

    for name, df in kwargs.items():

        print(name, '\n')

        print(f'Standard deviation:     {df.std()}')

        print(f'Skewness:               {df.skew()}')

        print(f'Kurtosis:               {df.kurtosis()}\n')
calculate_mop(Lyft= lyft['distance'], Uber= uber['distance'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

sns.distplot(lyft['distance'], ax=ax1, kde=True)

ax1.set_title('Distribution of distance in Lyft', fontsize=20)

ax1.set_ylim(0, 0.6)

a = sns.distplot(uber['distance'], ax=ax2)

ax2.set_title('Distribution of distance in Uber', fontsize=20)

ax2.set_ylim(0, 0.6)
# a.figure.savefig(r'C:\Users\gokul\Downloads\distance.jpg')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))

sns.boxplot(lyft['distance'], ax=ax1)

ax1.set_title('Lyft', fontsize=15)

ax1.set_xlim(0, 8)

sns.boxplot(uber['distance'], ax=ax2)

ax2.set_title('Uber', fontsize=15)
# lyft[lyft['distance']<0.3] # can we remove records below 0.3 as cancellation records
# uber[uber['distance']<0.25].sort_values(by='distance', ascending=False).head(30)
overall = cab_data['price'].describe()

overall # measure of central tendency
uber_price = uber['price'].describe()

uber_price
lyft[lyft.price<2.9].shape
lyft_price = lyft['price'].describe()

lyft_price
uber.price.sum(), lyft.price.sum()
df = pd.DataFrame({'Overall': overall.values,

                  'Lyft': lyft_price.values,

                  'Uber': uber_price.values}, index= ['Count', 'Mean', 'Std. Dev.', 'Min', '25%', '50%', '75%', 'Max'])

df
# df.to_csv(r'C:\Users\gokul\Downloads\metrics.csv')
calculate_mop(Lyft= lyft['price'], Uber= uber['price']) # measure of dispersion
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

a = sns.distplot(lyft['price'], ax=ax1)

ax1.set_title('Distribution of price in Lyft', fontsize=20)

ax1.set(xlabel='Price')

ax1.set_ylim(0, 0.12)

b =sns.distplot(uber[~uber['price'].isnull()]['price'], ax=ax2)

ax2.set_title('Distribution of price in Uber', fontsize=20)
# a.figure.savefig(r'C:\Users\gokul\Downloads\price.jpg')
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,5))

sns.boxplot(lyft['price'], ax=ax1)

ax1.set_title('Lyft', fontsize=15)



sns.boxplot(uber[~uber['price'].isnull()]['price'], ax=ax2)

ax2.set_title('Uber', fontsize=15)

ax2.set_xlim(0, 100)
lyft[(lyft['price']>40)].head(3)
uber[(uber['price']>40)].head(3)
# a = uber[uber['price']<40].groupby(by=['source', 'destination']).median()#.head(10)

# a
# a.to_csv(r'C:\Users\gokul\Downloads\tab.csv')
uber[uber['price']>40].groupby(by=['source', 'destination']).mean().head(10)
cab_data['cab_type'].value_counts() # frequency count
cab_data['cab_type'].value_counts(normalize=True) # percentage of values
plt.figure(figsize=(8,5))

sns.countplot('cab_type', data=cab_data)

plt.title('Frequency of Uber and Lyft data', fontsize=15)
lyft['name'].value_counts() # frequency count
uber['name'].value_counts()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

sns.countplot(lyft['name'], ax=ax1)

ax1.set_title('Frequency count of car models in Lyft', fontsize=20)

sns.countplot(uber['name'], ax=ax2)

ax2.set_title('Frequency count of car models  in Uber', fontsize=20)
lyft['source'].value_counts() # frequency count
uber['source'].value_counts()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

sns.countplot(lyft['source'], ax=ax1)

ax1.set_title('Frequency count of different source location in Lyft', fontsize=20)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=15)

# ax1.set_ylim(0, 25000)

sns.countplot(uber['source'], ax=ax2)

ax2.set_title('Frequency count of different source location  in Uber', fontsize=20)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=15)
lyft['destination'].value_counts()
uber['destination'].value_counts()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

sns.countplot(lyft['destination'], ax=ax1)

ax1.set_title('Frequency count of different destination location in Lyft', fontsize=20)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=15)

ax1.set_ylim(0, 30000)

sns.countplot(uber['destination'], ax=ax2)

ax2.set_title('Frequency count of different destination location  in Uber', fontsize=20)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=15)
lyft['product_id'].value_counts()
uber['product_id'].value_counts()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

sns.countplot(lyft['product_id'], ax=ax1)

ax1.set_title('Frequency count of different Product names in Lyft', fontsize=20)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=15)

sns.countplot(uber['product_id'], ax=ax2)

ax2.set_title('Frequency count of different Product names in Uber', fontsize=20)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=15)
lyft['surge_multiplier'].value_counts() # frequency count
uber['surge_multiplier'].value_counts()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

sns.countplot(lyft['surge_multiplier'], ax=ax1)

ax1.set_title('Frequency count of different surge multipliers in Lyft', fontsize=20)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=15)

ax1.set_ylim(0, 350000)

sns.countplot(uber['surge_multiplier'], ax=ax2)

ax2.set_title('Frequency count of different surge multipliers in Uber', fontsize=20)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=15)

ax2.set_ylim(0, 350000)
lyft['hour'].value_counts()
uber['hour'].value_counts()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

sns.countplot(lyft['hour'], ax=ax1)

ax1.set_title('Frequency count of different destination location in Lyft', fontsize=20)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=15)

ax1.set_ylim(0, 17500)

sns.countplot(uber['hour'], ax=ax2)

ax2.set_title('Frequency count of different destination location  in Uber', fontsize=20)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=15)

ax2.set_ylim(0, 17500)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 7))

sns.scatterplot(lyft['distance'], lyft['price'], ax=ax1)

ax1.set_title('Price vs Distance in Lyft', fontsize=20)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=15)

sns.scatterplot(uber['distance'], uber['price'], ax=ax2)

ax2.set_title('Price vs Distance in Uber', fontsize=20)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=15)

ax2.set_ylim(0, 100)
lyft['distance'].corr(lyft['price'])
uber['distance'].corr(uber['price'])
a= pd.crosstab(cab_data['cab_type'], cab_data['source'])

a
pd.crosstab(cab_data['cab_type'], cab_data['source'], normalize=True)
plt.figure(figsize=(10,7))

pd.crosstab(cab_data['cab_type'], cab_data['source']).plot.bar(stacked=True, figsize=(20,7), rot=0)
pd.crosstab(cab_data['cab_type'], cab_data['destination'])
pd.crosstab(cab_data['cab_type'], cab_data['destination'], normalize=True)
plt.figure(figsize=(10,7))

pd.crosstab(cab_data['cab_type'], cab_data['destination']).plot.bar(stacked=True, figsize=(20,7), rot=0)
pd.crosstab(cab_data['cab_type'], cab_data['surge_multiplier'])
pd.crosstab(cab_data['cab_type'], cab_data['surge_multiplier'], normalize=True)
plt.figure(figsize=(10,7))

pd.crosstab(cab_data['cab_type'], cab_data['surge_multiplier']).plot.bar(stacked=True, figsize=(20,7), rot=0)
pd.crosstab(lyft['name'], lyft['source'])
plt.figure(figsize=(10,7))

pd.crosstab(lyft['name'], lyft['source']).plot.bar(stacked=True, figsize=(20,7), rot=0)
cab_data['source'].nunique(), cab_data['destination'].nunique()
pd.crosstab(uber['name'], uber['source'])
pd.crosstab(lyft['name'], lyft['surge_multiplier'])
pd.crosstab(lyft['name'], lyft['surge_multiplier'], normalize=True)
plt.figure(figsize=(10,7))

pd.crosstab(lyft['name'], lyft['surge_multiplier']).plot.bar(stacked=True, figsize=(20,7), rot=0)
pd.crosstab(uber['name'], uber['surge_multiplier'])
pd.crosstab(uber['source'], uber['destination'])
weather_data.info() # basic info
# let's impute the unix epoch time to standard date time format

weather_data['time_stamp'] = pd.to_datetime(weather_data['time_stamp'], unit='s')

weather_data['date'] = weather_data['time_stamp'].dt.date

weather_data['hour'] = weather_data['time_stamp'].dt.hour



weather_data.drop('time_stamp', axis=1, inplace=True)



weather_data.head(3)
weather_data['temp'].describe()
calculate_mop(Temperature = weather_data['temp'])
plt.figure(figsize=(8,5))

sns.distplot(weather_data['temp'])

plt.title('Distribution of Temperature', fontsize=15)
plt.figure(figsize=(8,5))

sns.boxplot(weather_data['temp'])

plt.title('Distribution of Temperature', fontsize=15)
weather_data[weather_data.temp < 26].date.value_counts()
weather_data[weather_data.temp > 53].date.value_counts()
weather_data[weather_data['temp']<27].shape
weather_data[weather_data['temp']>52.5].shape
weather_data[weather_data.temp>53].location.value_counts()
weather_data['clouds'].describe()
calculate_mop(Clouds=weather_data['clouds'])
plt.figure(figsize=(8,5))

sns.distplot(weather_data['clouds'])

plt.title('Distribution of Clouds', fontsize=15)
plt.figure(figsize=(8,5))

sns.boxplot(weather_data['clouds'])

plt.title('Distribution of Clouds', fontsize=15)
weather_data['pressure'].describe()
calculate_mop(Pressure=weather_data['pressure'])
plt.figure(figsize=(8,5))

sns.distplot(weather_data['pressure'])

plt.title('Distribution of Pressure', fontsize=15)
plt.figure(figsize=(8,5))

sns.boxplot(weather_data['clouds'])

plt.title('Distribution of Clouds', fontsize=15)
weather_data['rain'].describe()
calculate_mop(Rain=weather_data['rain'])
plt.figure(figsize=(8,5))

a = sns.distplot(weather_data[~weather_data['rain'].isnull()]['rain'])

plt.title('Distribution of Rain', fontsize=15)
# a.figure.savefig(r'C:\Users\gokul\Downloads\rain.jpg')
plt.figure(figsize=(8,5))

sns.boxplot(weather_data[~weather_data['rain'].isnull()]['rain'])

plt.title('Distribution of Rain', fontsize=15)
weather_data[weather_data['rain']>0.13].date.value_counts()
weather_data['humidity'].describe()
calculate_mop(Humidity = weather_data['humidity'])
plt.figure(figsize=(8,5))

sns.distplot(weather_data['humidity'])

plt.title('Distribution of Humidity', fontsize=15)
plt.figure(figsize=(8,5))

sns.boxplot(weather_data['humidity'])

plt.title('Distribution of Humidity', fontsize=15)
weather_data['wind'].describe()
calculate_mop(Wind=weather_data['wind'])
plt.figure(figsize=(8,5))

sns.distplot(weather_data['wind'])

plt.title('Distribution of Wind', fontsize=15)
plt.figure(figsize=(8,5))

sns.boxplot(weather_data['wind'])

plt.title('Distribution of Wind', fontsize=15)
weather_data['location'].value_counts()
sns.pairplot(weather_data[['temp', 'clouds', 'rain', 'pressure', 'humidity', 'wind']])
sns.boxplot(weather_data['temp'], y=weather_data['location'], orient='h')
sns.boxplot(weather_data['rain'], y=weather_data['location'], orient='h')
sns.boxplot(weather_data['clouds'], y=weather_data['location'], orient='h')
sns.boxplot(weather_data['pressure'], y=weather_data['location'], orient='h')
sns.boxplot(weather_data['humidity'], y=weather_data['location'], orient='h')
sns.boxplot(weather_data['wind'], y=weather_data['location'], orient='h')
weather_data['date'].value_counts().sort_index()
nrows, ncols = cab_data.shape

print(f'Cab ride dataset contains {nrows} rows and {ncols} columns.')
mv  = cab_data.isnull().sum().sum()

prop = round(((mv/cab_data.shape[0]) * 100),3)

print(f'Cab ride dataset contains {mv} missing values, which is {prop} % of whole data.')
cab_data.isnull().sum()
# let's check the cab type

cab_data[cab_data['price'].isnull()]['cab_type'].value_counts() 
cab_data[cab_data['cab_type']=='Uber'].name.value_counts()
cab_data[cab_data['price'].isnull()]['name'].value_counts() # car model
# let's drop those records

cab_data.dropna(how='any', inplace=True)

nrows, ncols = cab_data.shape

print(f'Now the dataset contains {nrows} rows and {ncols} columns.')



uber = cab_data[cab_data['cab_type']=='Uber']

lyft = cab_data[cab_data['cab_type']=='Lyft']
cab_data.isnull().sum().sum() # check for missing values
# cab_data.to_csv('C:\Users\gokul\Downloads\cabs.csv')
nrows, ncols = weather_data.shape

print(f'Cab ride dataset contains {nrows} rows and {ncols} columns.')
mv  = weather_data.isnull().sum().sum()

prop = round(((mv/weather_data.shape[0]) * 100),3)

print(f'Cab ride dataset contains {mv} missing values, which is {prop} % of whole data.')
weather_data.isnull().sum()
# let's impute the missing values in the 'rain' column with 0

weather_data['rain'].fillna(0, inplace=True)
weather_data.isnull().sum().sum() # check for missing values
# weather data supposed to contain 1 record per hour, since it has more than one values for few hours, 

# we took groupby average

weather_data = weather_data.groupby(['location','date', 'hour']).mean()

weather_data.reset_index(inplace=True)
merged_data = pd.merge(cab_data, weather_data, how='left', left_on=['source', 'date', 'hour'],

        right_on=['location', 'date', 'hour'])
merged_data.info()
merged_data[merged_data.temp.isnull()].groupby(['source', 'date', 'hour']).mean().head(6)
df1 = weather_data.loc[

    (weather_data['date']==datetime.date(2018, 11, 28)) &

    (weather_data['hour']==0)]



df2 = weather_data.loc[

    (weather_data['date']==datetime.date(2018, 12, 4)) &

    (weather_data['hour']==5)]

df3 = weather_data.loc[

    (weather_data['date']==datetime.date(2018, 11, 28)) &

    (weather_data['hour']==2)]

df4 = weather_data.loc[

    (weather_data['date']==datetime.date(2018, 12, 4)) &

    (weather_data['hour']==7)]





lookup = pd.concat([df1, df2, df3, df4])

lookup = lookup.groupby(['hour', 'location', 'date']).mean().reset_index()

df5 = weather_data.loc[

    (weather_data['date']==datetime.date(2018, 12, 18)) &

    (weather_data['hour']==18)]



lookup = pd.concat([lookup, df5])

lookup['hour'] += 1

lookup.reset_index(inplace=True)
weather_data = pd.concat([weather_data, lookup], ignore_index=True) 
weather_data.shape
cab_data = pd.merge(cab_data, weather_data, how='left',

                left_on=['source', 'date', 'hour'],

                right_on=['location', 'date', 'hour'])
cab_data.info()
cab_data.drop('index', axis=1, inplace=True)
cab_data.shape, cab_data.drop_duplicates().shape
# drop unnecessary features

cab_data = cab_data.drop(['id', 'product_id', 'location', 'date'], axis=1)
corr_m = cab_data.corr()
x = np.tri(corr_m.shape[0],k=-1)
plt.figure(figsize=(15,10))

a = sns.heatmap(corr_m, annot=True, mask=x)
# a.figure.savefig(r'C:\Users\gokul\Downloads\corr.jpg')
# Initial data preparation
data = cab_data.drop(['price', 'surge_multiplier'], axis=1) # we are dropping surge multiplier, to avoid data leak

labels = cab_data['price'].copy()
# model building libraries



# from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder
uber = cab_data[cab_data['cab_type']=='Uber']

uber.reset_index(inplace=True)

uber.drop('index', axis=1, inplace=True)

lyft = cab_data[cab_data['cab_type']=='Lyft']

lyft.reset_index(inplace=True)

lyft.drop('index', axis=1, inplace=True)
uber.drop('cab_type', axis=1, inplace=True)

lyft.drop('cab_type', axis=1, inplace=True)
lyft_data = lyft.copy() # backups

uber_data = uber.copy()
uber_data.head()
uber.info()
ohe = OneHotEncoder()

car_type = pd.DataFrame(ohe.fit_transform(uber[['name']]).toarray(), columns=sorted(list(uber['name'].unique())))

source = pd.DataFrame(ohe.fit_transform(uber[['source']]).toarray(), 

                       columns=['src_'+loc for loc in sorted(list(uber['source'].unique()))])

destination = pd.DataFrame(ohe.fit_transform(uber[['destination']]).toarray(), 

                           columns=['dest_'+loc for loc in sorted(list(uber['destination'].unique()))])
ohe = OneHotEncoder()

lyft_car_type = pd.DataFrame(ohe.fit_transform(lyft[['name']]).toarray(), columns=sorted(list(lyft['name'].unique())))

lyft_source = pd.DataFrame(ohe.fit_transform(lyft[['source']]).toarray(),

                           columns=['src_'+loc for loc in sorted(list(lyft['source'].unique()))])

lyft_destination = pd.DataFrame(ohe.fit_transform(lyft[['destination']]).toarray(),

                                columns=['dest_'+loc for loc in sorted(list(lyft['destination'].unique()))])
uber = pd.concat([uber, car_type, source, destination], axis=1)

uber.drop(['name', 'source', 'destination'], axis=1, inplace=True)
lyft = pd.concat([lyft, lyft_car_type, lyft_source, lyft_destination], axis=1)

lyft.drop(['name', 'source', 'destination'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder
uber_le = uber_data.copy()

lyft_le = lyft_data.copy()



lb = LabelEncoder()



uber_le['name'] = lb.fit_transform(uber_data['name'])

uber_le['source'] = lb.fit_transform(uber_data['source'])

uber_le['destination'] = lb.fit_transform(uber_data['destination'])



lyft_le['name'] = lb.fit_transform(lyft_le['name'])

lyft_le['source'] = lb.fit_transform(lyft_le['source'])

lyft_le['destination'] = lb.fit_transform(lyft_le['destination'])
uber_leX = uber_le.drop(['price', 'surge_multiplier'], axis=1)

uber_ley = uber_le['price'].copy()



lyft_leX = lyft_le.drop(['price', 'surge_multiplier'], axis=1)

lyft_ley = lyft_le['price'].copy()
uber_X = uber.drop(['price', 'surge_multiplier'], axis=1)

uber_y = uber['price'].copy()
lyft_X = lyft.drop(['price', 'surge_multiplier'], axis=1)

lyft_y = lyft['price'].copy()
uber_leX.shape
lyft_leX.shape
import statsmodels.api as sm
x_constant = sm.add_constant(uber_X)

uber_model = sm.OLS(uber_y, x_constant).fit()

uber_model.summary()
x_constant = sm.add_constant(lyft_X)

lyft_model = sm.OLS(lyft_y, x_constant).fit()

lyft_model.summary()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
X_train, X_test, y_train, y_test = train_test_split(uber_X, uber_y, test_size=0.3, random_state=42)



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
train_pred = lin_reg.predict(X_train)

print(f'Train score {np.sqrt(mean_squared_error(y_train, train_pred))}')



predicted = lin_reg.predict(X_test)

print(f'Test score {np.sqrt(mean_squared_error(y_test, predicted))}')
X_train, X_test, y_train, y_test = train_test_split(lyft_X, lyft_y, test_size=0.3, random_state=42)



lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
train_pred = lin_reg.predict(X_train)

print(f'Train score {np.sqrt(mean_squared_error(y_train, train_pred))}')



predicted = lin_reg.predict(X_test)

print(f'Test score {np.sqrt(mean_squared_error(y_test, predicted))}')
import statsmodels.api         as     sm

from   statsmodels.formula.api import ols

 

mod = ols('price ~ source', data = uber_data).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
mod = ols('price ~ source', data = lyft_data).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
mod = ols('price ~ destination', data = uber_data).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
mod = ols('price ~ destination', data = lyft_data).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
mod = ols('price ~ name', data = uber_data).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
mod = ols('price ~ name', data = lyft_data).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
mod = ols('price ~ cab_type', data = cab_data).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

print(aov_table)
lyft_data.info()
plt.figure(figsize=(15,10))

corr_m = lyft_data.corr()

x = np.tri(corr_m.shape[0],k=-1)

sns.heatmap(corr_m, annot=True, cmap=plt.cm.Reds, mask=x)

plt.show()
corr_m['price'].abs().sort_values(ascending=False)[1:]
plt.figure(figsize=(15,10))

corr_m = uber_data[['distance', 'destination', 'source', 'price','name', 'hour', 'temp', 'clouds', 'pressure', 'rain', 'humidity',

       'wind']].corr()

x = np.tri(corr_m.shape[0],k=-1)

sns.heatmap(corr_m, annot=True, cmap=plt.cm.Reds, mask=x)

plt.show()
corr_m['price'].abs().sort_values(ascending=False)[1:]
uber1_X = uber_X.copy()

uber1_y = uber_y.copy()
lyft1_X = lyft_X

lyft1_y = lyft_y
#Backward Elimination

cols = list(uber1_X.columns)

pmax = 1

counter=0

while (len(cols)>0):

    p= []

    counter+=1



    X_1 = uber1_X[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(uber1_y,X_1).fit()

#     print(counter)

#     print(len(pd.Series(model.pvalues.values)))

    p = pd.Series(model.pvalues.values[1:],index = cols)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

#         print('inside')

        cols.remove(feature_with_p_max)

    else:

        break

    print(feature_with_p_max)

#     print(len(cols))

selected_features_BE = cols

print(selected_features_BE)
len(selected_features_BE)
uber2 = uber1_X[selected_features_BE]
uber2_X = uber2

uber2_y = uber_data['price'].copy()
x_constant = sm.add_constant(uber2_X)

uber_model = sm.OLS(uber2_y, x_constant).fit()

uber_model.summary()
#Backward Elimination

cols = list(lyft1_X.columns)

pmax = 1

counter=0

while (len(cols)>0):

    p= []

    counter+=1

    X_1 = lyft1_X[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(lyft1_y,X_1).fit()

#     print(counter)

#     print(len(pd.Series(model.pvalues.values)))

    p = pd.Series(model.pvalues.values[1:],index = cols)      

    pmax = max(p)

    feature_with_p_max = p.idxmax()

    if(pmax>0.05):

#         print('inside')

        cols.remove(feature_with_p_max)

    else:

        break

    print(feature_with_p_max)

#     print(len(cols))

selected_features_BE = cols

print('\n',selected_features_BE)
len(selected_features_BE)
lyft2 = lyft1_X[selected_features_BE]
lyft2_X = lyft2

lyft2_y = lyft_data['price'].copy()
x_constant = sm.add_constant(lyft2_X)

lyft_model = sm.OLS(lyft2_y, x_constant).fit()

lyft_model.summary()
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
# Build RF classifier to use in feature selection

clf = LinearRegression()



X_train, X_test, y_train, y_test = train_test_split(uber1_X, uber1_y, test_size = 0.3, random_state = 0)





# Build step forward feature selection

sfs1 = sfs(clf,k_features = 38,forward=True,

           floating=False, scoring='r2',

           verbose=2,

           cv=5)



# Perform SFFS

sfs1 = sfs1.fit(X_train, y_train)
# Build RF classifier to use in feature selection

clf = LinearRegression()



X_train, X_test, y_train, y_test = train_test_split(lyft1_X, lyft1_y, test_size = 0.3, random_state = 0)





# Build step forward feature selection

sfs1 = sfs(clf,k_features = 38,forward=True,

           floating=False, scoring='r2',

           verbose=2,

           cv=5)



# Perform SFFS

sfs1 = sfs1.fit(X_train, y_train)
from sklearn.linear_model import LassoCV
reg = LassoCV()

reg.fit(uber1_X, uber1_y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(uber1_X,uber1_y))

coef = pd.Series(reg.coef_, index = uber1_X.columns)

coef
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
reg = LassoCV()

reg.fit(lyft1_X, lyft1_y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(lyft1_X, lyft1_y))

coef = pd.Series(reg.coef_, index = lyft1_X.columns)

coef
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
## Building of simple OLS model.

X_constant = sm.add_constant(uber1_X)

model = sm.OLS(uber1_y, X_constant).fit()

predictions = model.predict(X_constant)

model.summary()
### calculating the vif values as multicollinearity exists



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = [variance_inflation_factor(uber1_X.values, j) for j in range(1, uber1_X.shape[1])]

vif
# removing collinear variables

# function definition



def calculate_vif(x):

    thresh = 5.0

    output = pd.DataFrame()

    k = x.shape[1]

    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]

    for i in range(1,k):

        print("Iteration no.")

        print(i)

        print(vif)

        a = np.argmax(vif)

        print("Max VIF is for variable no.:")

        print(a)

        

        if vif[a] <= thresh :

            break

        if i == 1 :          

            output = x.drop(x.columns[a], axis = 1)

            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]

        elif i > 1 :

            output = output.drop(output.columns[a],axis = 1)

            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]

        print(output.columns)

    return(output)
## passing X to the function so that the multicollinearity gets removed.

train_out = calculate_vif(uber1_X)
## includes only the relevant features.

train_out.head()
len(train_out.columns)
uber_X = uber_X.drop(['wind', 'humidity', 'temp', 'clouds'], axis=1) #onehot encoded

lyft_X = lyft_X.drop(['wind', 'humidity', 'temp', 'clouds'], axis=1)
uber_leX = uber_leX.drop(['wind', 'humidity', 'temp', 'clouds'], axis=1) # label encoded

lyft_leX = lyft_leX.drop(['wind', 'humidity', 'temp', 'clouds'], axis=1)
uber_leX.head()
lyft_leX.head()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



uber_std = pd.DataFrame(sc.fit_transform(uber_X[['distance', 'hour', 'pressure', 'rain']]), 

                        columns=['distance', 'hour', 'pressure', 'rain'])



lyft_std = pd.DataFrame(sc.fit_transform(lyft_X[['distance', 'hour', 'pressure', 'rain']]),

                        columns=['distance', 'hour', 'pressure', 'rain'])



uber_X = uber_X.drop(['distance', 'hour', 'pressure', 'rain'], axis=1)

lyft_X = lyft_X.drop(['distance', 'hour', 'pressure', 'rain'], axis=1)



uber_X = pd.concat([uber_std, uber_X], axis=1)

lyft_X = pd.concat([lyft_std, lyft_X], axis=1)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV
X_trainu, X_testu, y_trainu, y_testu = train_test_split(uber_X, uber_y, test_size=0.3, random_state=42)
lin_reg_uber = LinearRegression()

lin_reg_uber.fit(X_trainu, y_trainu)



# print(f'Train score : {lin_reg_uber.score(X_trainu, y_trainu)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, lin_reg_uber.predict(X_trainu)))}')

predicted = lin_reg_uber.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')
train_cv = cross_val_score(LinearRegression(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(LinearRegression(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



l_reg_uber = {}

l_reg_uber['Train'] = round(train_rmse, 4)

l_reg_uber['Test'] = round(test_rmse, 4)

l_reg_uber
X_trainl, X_testl, y_trainl, y_testl = train_test_split(lyft_X, lyft_y, test_size=0.3, random_state=42)
lin_reg_lyft = LinearRegression()

lin_reg_lyft.fit(X_trainl, y_trainl)



# print(f'Train score : {lin_reg_lyft.score(X_trainl, y_trainl)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, lin_reg_lyft.predict(X_trainl)))}')

predicted = lin_reg_lyft.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')
train_cv = cross_val_score(LinearRegression(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(LinearRegression(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



l_reg_lyft = {}

l_reg_lyft['Train'] = round(train_rmse, 4)

l_reg_lyft['Test'] = round(test_rmse, 4)

l_reg_lyft
ridge_reg = Ridge(random_state=42)

ridge_reg.fit(X_trainu, y_trainu)



ridge_reg_predict = ridge_reg.predict(X_testu)



# print(f'Train score : {ridge_reg.score(X_trainu, y_trainu)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, ridge_reg.predict(X_trainu)))}')

predicted = ridge_reg.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



np.sqrt(np.abs(cross_val_score(Ridge(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')))
train_cv = cross_val_score(Ridge(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(Ridge(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



r_reg_uber = {}

r_reg_uber['Train'] = round(train_rmse, 4)

r_reg_uber['Test'] = round(test_rmse, 4)

r_reg_uber
lambdas=np.linspace(1,100,100)

params={'alpha':lambdas}

grid_search=GridSearchCV(Ridge(),param_grid=params,cv=10,scoring='neg_mean_absolute_error')

grid_search.fit(X_trainu,y_trainu)

grid_search.best_estimator_
model = grid_search.best_estimator_



# print(f'Train score : {model.score(X_trainu, y_trainu)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, model.predict(X_trainu)))}')

predicted = model.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



# cross_val_score(model, X_trainu, y_trainu, cv=5, n_jobs=-1)
ridge_reg = Ridge(random_state=42)

ridge_reg.fit(X_trainl, y_trainl)



ridge_reg_predict = ridge_reg.predict(X_testl)



# print(f'Train score : {ridge_reg.score(X_trainl, y_trainl)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, ridge_reg.predict(X_trainl)))}')

predicted = ridge_reg.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



# cross_val_score(Ridge(), X_trainl, y_trainl, cv=5, n_jobs=-1)
train_cv = cross_val_score(Ridge(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(Ridge(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



r_reg_lyft = {}

r_reg_lyft['Train'] = round(train_rmse, 4)

r_reg_lyft['Test'] = round(test_rmse, 4)

r_reg_lyft
lambdas=np.linspace(1,100,100)

params={'alpha':lambdas}

grid_search=GridSearchCV(Ridge(),param_grid=params,cv=10,scoring='neg_mean_absolute_error')

grid_search.fit(X_trainl,y_trainl)

grid_search.best_estimator_
model = grid_search.best_estimator_



# print(f'Train score : {model.score(X_trainl, y_trainl)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, model.predict(X_trainl)))}')

predicted = model.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



# cross_val_score(model, X_trainl, y_trainl, cv=5, n_jobs=-1)
lasso_reg = Lasso(random_state=42)

lasso_reg.fit(X_trainu, y_trainu)



lasso_reg_predict = lasso_reg.predict(X_testu)



# print(f'Train score : {lasso_reg.score(X_trainu, y_trainu)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, lasso_reg.predict(X_trainu)))}')

predicted = lasso_reg.predict(X_testu)

# print(np.sqrt(mean_squared_error(y_trainu, lasso_reg.predict(X_trainu))))

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



# cross_val_score(Lasso(), X_trainu, y_trainu, cv=5, n_jobs=-1)
train_cv = cross_val_score(Lasso(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(Lasso(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



la_reg_uber = {}

la_reg_uber['Train'] = round(train_rmse, 4)

la_reg_uber['Test'] = round(test_rmse, 4)

la_reg_uber
lambdas=np.linspace(1,100,100)

params={'alpha':lambdas}

grid_search=GridSearchCV(Lasso(),param_grid=params,cv=10,scoring='neg_mean_absolute_error')

grid_search.fit(X_trainu,y_trainu)

grid_search.best_estimator_
model = grid_search.best_estimator_



# print(f'Train score : {model.score(X_trainu, y_trainu)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, model.predict(X_trainu)))}')

predicted = model.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



# cross_val_score(model, X_trainu, y_trainu, cv=5, n_jobs=-1)
lasso_reg = Lasso(random_state=42)

lasso_reg.fit(X_trainl, y_trainl)



# print(f'Train score : {lasso_reg.score(X_trainl, y_trainl)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, lasso_reg.predict(X_trainl)))}')

predicted = lasso_reg.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



# cross_val_score(Lasso(random_state=42), X_trainl, y_trainl, cv=5, n_jobs=-1)
train_cv = cross_val_score(Lasso(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(Lasso(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



la_reg_lyft = {}

la_reg_lyft['Train'] = round(train_rmse, 4)

la_reg_lyft['Test'] = round(test_rmse, 4)

la_reg_lyft
lambdas=np.linspace(1,100,100)

params={'alpha':lambdas}

grid_search=GridSearchCV(Lasso(),param_grid=params,cv=10,scoring='neg_mean_absolute_error')

grid_search.fit(X_trainl,y_trainl)

grid_search.best_estimator_
model = grid_search.best_estimator_



# print(f'Train score : {model.score(X_trainl, y_trainl)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, model.predict(X_trainl)))}')

predicted = model.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



np.sqrt(np.abs(cross_val_score(model, X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_absolute_error')))
elastic_reg = ElasticNet(random_state=42)

elastic_reg.fit(X_trainu, y_trainu)



elastic_reg_predict = elastic_reg.predict(X_testu)



# print(f'Train score : {elastic_reg.score(X_trainu, y_trainu)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, elastic_reg.predict(X_trainu)))}')

predicted = elastic_reg.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



# cross_val_score(ElasticNet(), X_trainu, y_trainu, cv=5, n_jobs=-1)
train_cv = cross_val_score(ElasticNet(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(ElasticNet(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



el_reg_uber = {}

el_reg_uber['Train'] = round(train_rmse, 4)

el_reg_uber['Test'] = round(test_rmse, 4)

el_reg_uber
# parametersGrid = {"alpha": [ 0.001, 0.01, 0.1, 1, 10, 100],

#                   "l1_ratio": np.arange(0.2, 1.0, 0.1)}

params={'alpha':lambdas}



grid_search=GridSearchCV(ElasticNet(),param_grid=params,cv=10,scoring='r2')

grid_search.fit(X_trainu,y_trainu)

grid_search.best_estimator_
model = grid_search.best_estimator_



# print(f'Train score : {model.score(X_trainu, y_trainu)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainu, model.predict(X_trainu)))}')

predicted = model.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



cross_val_score(model, X_trainu, y_trainu, cv=5, n_jobs=-1)
elastic_reg = ElasticNet(random_state=42)

elastic_reg.fit(X_trainl, y_trainl)



elastic_reg_predict = elastic_reg.predict(X_testl)



# print(f'Train score : {elastic_reg.score(X_trainl, y_trainl)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, elastic_reg.predict(X_trainl)))}')

predicted = elastic_reg.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



cross_val_score(ElasticNet(), X_trainl, y_trainl, cv=5, n_jobs=-1)
train_cv = cross_val_score(ElasticNet(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(ElasticNet(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



el_reg_lyft = {}

el_reg_lyft['Train'] = round(train_rmse, 4)

el_reg_lyft['Test'] = round(test_rmse, 4)

el_reg_lyft
# parametersGrid = {"alpha": [ 0.001, 0.01, 0.1, 1, 10, 100],

#                   "l1_ratio": np.arange(0.2, 1.0, 0.1)}

params={'alpha':lambdas}



grid_search=GridSearchCV(ElasticNet(),param_grid=params,cv=10,scoring='r2')

grid_search.fit(X_trainl,y_trainl)

grid_search.best_estimator_
model = grid_search.best_estimator_



# print(f'Train score : {model.score(X_trainl, y_trainl)}')

print(f'Train RMSE score : {np.sqrt(mean_squared_error(y_trainl, model.predict(X_trainl)))}')

predicted = model.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



# cross_val_score(model, X_trainl, y_trainl, cv=5, n_jobs=-1)
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import roc_curve
X_trainu, X_testu, y_trainu, y_testu = train_test_split(uber_leX, uber_ley, test_size=0.3, random_state=42)
dtree = DecisionTreeRegressor()



dtree.fit(X_trainu, y_trainu)



train_pred = dtree.predict(X_trainu)



tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))

print(f'Train score : {tr_rmse}')

predicted = dtree.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



# cross_val_score(DecisionTreeRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1)
max_depth = range(1,20)

train_results = []

test_results = []

for n in max_depth:

    dt = DecisionTreeRegressor(max_depth=n)

    dt.fit(X_trainu, y_trainu)

    train_pred = dt.predict(X_trainu)

    rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))

    train_results.append(rmse)

    y_pred = dt.predict(X_testu)

    ts_rmse = np.sqrt(mean_squared_error(y_testu, y_pred))

    test_results.append(ts_rmse)
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depth, train_results, 'b', label='Train RMSE')

line2, = plt.plot(max_depth, test_results, 'r--', label='Test RMSE')

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('RMSE score')

plt.xlabel('Tree depth')

plt.show()
dtree = DecisionTreeRegressor(max_depth=15)



dtree.fit(X_trainu, y_trainu)



train_pred = dtree.predict(X_trainu)



tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))

print(f'Train score : {tr_rmse}')

predicted = dtree.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



# cross_val_score(DecisionTreeRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1)
train_cv = cross_val_score(DecisionTreeRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(DecisionTreeRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



dt_reg_uber = {}

dt_reg_uber['Train'] = round(train_rmse, 4)

dt_reg_uber['Test'] = round(test_rmse, 4)

dt_reg_uber
X_trainl, X_testl, y_trainl, y_testl = train_test_split(lyft_leX, lyft_ley, test_size=0.3, random_state=42)
dtree = DecisionTreeRegressor(max_depth=15)



dtree.fit(X_trainl, y_trainl)



train_pred = dtree.predict(X_trainl)



tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))

print(f'Train score : {tr_rmse}')

predicted = dtree.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



# cross_val_score(DecisionTreeRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1)
train_cv = cross_val_score(DecisionTreeRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(DecisionTreeRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



dt_reg_lyft = {}

dt_reg_lyft['Train'] = round(train_rmse, 4)

dt_reg_lyft['Test'] = round(test_rmse, 4)

dt_reg_lyft
param_grid = {'max_depth': np.arange(3, 30),

             'min_samples_split': np.arange(.1,1.1,.1),

             'min_samples_leaf': np.arange(.1,.6,.1)}
grid_srch_dtree = tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10,scoring='neg_mean_squared_error')

grid_srch_dtree.fit(X_trainu, y_trainu)

grid_srch_dtree.best_estimator_
from sklearn.ensemble import RandomForestRegressor

# from sklearn.cross
rf = RandomForestRegressor()

rf.fit(X_trainu, y_trainu)



train_pred = rf.predict(X_trainu)



tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))

print(f'Train score : {tr_rmse}')

predicted = rf.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



cv = cross_val_score(RandomForestRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(cv)))
train_cv = cross_val_score(RandomForestRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(RandomForestRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



rf_reg_uber = {}

rf_reg_uber['Train'] = round(train_rmse, 4)

rf_reg_uber['Test'] = round(test_rmse, 4)

rf_reg_uber
param_grid = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],

              'max_features' : list(range(1,X_trainu.shape[1])),

              'max_depth': np.arange(3, 30),

             'min_samples_split': np.arange(.1,1.1,.1),

             'min_samples_leaf': np.arange(.1,.6,.1)}
grid_srch_rf = tree = GridSearchCV(RandomForestRegressor(), param_grid, cv=10,scoring='neg_mean_squared_error')

grid_srch_rf.fit(X_trainu, y_trainu)

grid_srch_rf.best_estimator_
rf = RandomForestRegressor()

rf.fit(X_trainl, y_trainl)



train_pred = rf.predict(X_trainl)



tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))

print(f'Train score : {tr_rmse}')

predicted = rf.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



cv = cross_val_score(RandomForestRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(cv)))
train_cv = cross_val_score(RandomForestRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(RandomForestRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



rf_reg_lyft = {}

rf_reg_lyft['Train'] = round(train_rmse, 4)

rf_reg_lyft['Test'] = round(test_rmse, 4)

rf_reg_lyft
from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor(random_state=42)



abr.fit(X_trainu, y_trainu)



train_pred = abr.predict(X_trainu)



tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))

print(f'Train score : {tr_rmse}')

predicted = abr.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



cv = cross_val_score(AdaBoostRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(cv)))
train_cv = cross_val_score(AdaBoostRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(AdaBoostRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



abr_reg_uber = {}

abr_reg_uber['Train'] = round(train_rmse, 4)

abr_reg_uber['Test'] = round(test_rmse, 4)

abr_reg_uber
abr = AdaBoostRegressor(random_state=42)



abr.fit(X_trainl, y_trainl)



train_pred = abr.predict(X_trainl)



tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))

print(f'Train score : {tr_rmse}')

predicted = abr.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



cv = cross_val_score(AdaBoostRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(cv)))
train_cv = cross_val_score(AdaBoostRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(AdaBoostRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



abr_reg_lyft = {}

abr_reg_lyft['Train'] = round(train_rmse, 4)

abr_reg_lyft['Test'] = round(test_rmse, 4)

abr_reg_lyft
from sklearn.ensemble import GradientBoostingRegressor
X_trainu.head()
gbr = GradientBoostingRegressor(random_state=42)



gbr.fit(X_trainu, y_trainu)



train_pred = gbr.predict(X_trainu)



tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))

print(f'Train score : {tr_rmse}')

predicted = gbr.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



cv = cross_val_score(GradientBoostingRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(cv)))
train_cv = cross_val_score(GradientBoostingRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(GradientBoostingRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



gbr_reg_uber = {}

gbr_reg_uber['Train'] = round(train_rmse, 4)

gbr_reg_uber['Test'] = round(test_rmse, 4)

gbr_reg_uber
gbr = GradientBoostingRegressor(random_state=42)



gbr.fit(X_trainl, y_trainl)



train_pred = gbr.predict(X_trainl)



tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))

print(f'Train score : {tr_rmse}')

predicted = gbr.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



cv = cross_val_score(GradientBoostingRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(cv)))
train_cv = cross_val_score(GradientBoostingRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(GradientBoostingRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



gbr_reg_lyft = {}

gbr_reg_lyft['Train'] = round(train_rmse, 4)

gbr_reg_lyft['Test'] = round(test_rmse, 4)

gbr_reg_lyft
from xgboost import XGBRegressor
xbr = XGBRegressor(random_state=42)



xbr.fit(X_trainu, y_trainu)



train_pred = xbr.predict(X_trainu)



tr_rmse = np.sqrt(mean_squared_error(y_trainu, train_pred))

print(f'Train score : {tr_rmse}')

predicted = xbr.predict(X_testu)

rmse = np.sqrt(mean_squared_error(y_testu, predicted))

print(f'Test score : {rmse}')



cv = cross_val_score(XGBRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(cv)))
train_cv = cross_val_score(XGBRegressor(), X_trainu, y_trainu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(XGBRegressor(), X_testu, y_testu, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



xbr_reg_uber = {}

xbr_reg_uber['Train'] = round(train_rmse, 4)

xbr_reg_uber['Test'] = round(test_rmse, 4)

xbr_reg_uber
xbr = XGBRegressor(random_state=42)



xbr.fit(X_trainl, y_trainl)



train_pred = xbr.predict(X_trainl)



tr_rmse = np.sqrt(mean_squared_error(y_trainl, train_pred))

print(f'Train score : {tr_rmse}')

predicted = xbr.predict(X_testl)

rmse = np.sqrt(mean_squared_error(y_testl, predicted))

print(f'Test score : {rmse}')



cv = cross_val_score(XGBRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

print(np.sqrt(np.abs(cv)))
train_cv = cross_val_score(XGBRegressor(), X_trainl, y_trainl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(XGBRegressor(), X_testl, y_testl, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



xbr_reg_lyft = {}

xbr_reg_lyft['Train'] = round(train_rmse, 4)

xbr_reg_lyft['Test'] = round(test_rmse, 4)

xbr_reg_lyft
from catboost import CatBoostRegressor
X = uber_data.drop(['surge_multiplier', 'price', 'humidity', 'clouds', 'temp', 'wind'], axis=1)

y = uber_data['price'].copy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



categorical_features_indices = np.where(X.dtypes != np.number)[0]

categorical_features_indices
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE', verbose=400)

model.fit(X_train, y_train,cat_features=[1,2,3,4],eval_set=(X_test, y_test),plot=True)
train_cv = cross_val_score(CatBoostRegressor(), X_train, y_train, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(CatBoostRegressor(), X_test, y_test, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



cbr_reg_uber = {}

cbr_reg_uber['Train'] = round(train_rmse, 4)

cbr_reg_uber['Test'] = round(test_rmse, 4)

cbr_reg_uber
X = lyft_data.drop(['surge_multiplier', 'price', 'humidity', 'clouds', 'temp', 'wind'], axis=1)

y = lyft_data['price'].copy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



categorical_features_indices = np.where(X.dtypes != np.number)[0]

categorical_features_indices
model=CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')

model.fit(X_train, y_train,cat_features=[1,2,3,4],eval_set=(X_test, y_test),plot=True)
train_cv = cross_val_score(CatBoostRegressor(), X_train, y_train, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

train_rmse = np.sqrt(np.abs(train_cv)).mean()



test_cv = cross_val_score(CatBoostRegressor(), X_test, y_test, cv=5, n_jobs=-1,scoring='neg_mean_squared_error')

test_rmse = np.sqrt(np.abs(test_cv)).mean()



xbr_reg_lyft = {}

xbr_reg_lyft['Train'] = round(train_rmse, 4)

xbr_reg_lyft['Test'] = round(test_rmse, 4)

xbr_reg_lyft
final_results = pd.DataFrame([l_reg_uber, r_reg_uber, la_reg_uber, el_reg_uber, dt_reg_uber,

                              rf_reg_uber, abr_reg_uber, gbr_reg_uber, xbr_reg_uber, cbr_reg_uber],

                            index=['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression',

                                  'Decision Tree', 'Random Forest', 'Ada Boost', 'Gradient Boost', 'Xg Boost',

                                  'Cat Boost'])

final_results
final_results = pd.DataFrame([l_reg_lyft, r_reg_lyft, la_reg_lyft, el_reg_lyft, dt_reg_lyft,

                              rf_reg_lyft, abr_reg_lyft, gbr_reg_lyft, xbr_reg_lyft, cbr_reg_lyft],

                            index=['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net Regression',

                                  'Decision Tree', 'Random Forest', 'Ada Boost', 'Gradient Boost', 'Xg Boost',

                                  'Cat Boost'])

final_results
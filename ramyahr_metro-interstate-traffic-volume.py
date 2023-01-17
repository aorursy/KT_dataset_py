#Import necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
#Importing dataset from csv to data frame

df_traffic_data = pd.read_csv('../input/Metro_Interstate_Traffic_Volume.csv')
df_traffic_data.head()
df_traffic_data.shape
df_traffic_data.dtypes
df_traffic_data.info()
df_traffic_data.describe()
df_traffic_data.describe(include='object')
print("max date :" +df_traffic_data.date_time.max())

print("min date :" +df_traffic_data.date_time.min())
#Plotting frequency of each category in holiday column

plt.figure(figsize = (8,6))

sns.countplot(y='holiday', data = df_traffic_data)

plt.show()
#'None' is far greater than the other days. Removing None data to visualize the others

holidays = df_traffic_data.loc[df_traffic_data.holiday != 'None']

plt.figure(figsize=(8,6))

sns.countplot(y='holiday', data= holidays)

plt.show()
#plotting distribution of temperature variable

plt.figure(figsize=(6,4))

sns.boxplot('temp', data = df_traffic_data)

plt.show()
#Temperature is measured in Kelvin, changing to degree celsius to make it more intuitive

#convert kelvin to celsius

#(0K − 273.15)

df_traffic_data['temp'] = (df_traffic_data['temp']-273.15)

plt.figure(figsize=(6,4))

sns.boxplot('temp', data = df_traffic_data)

plt.show()
#There is one data point far away from the rest around -300 degrees celsius. Clearly, this is an error in recording.

#Eliminating will be eliminated in the data cleaning phase.
#Plotting rain variable

plt.figure(figsize=(6,4))

sns.distplot(df_traffic_data.rain_1h)

plt.show()

#From the distribution, it shows that the data is extremely skewed. Most of the observations are concentrated around 0.
#Plotting observations with values less than 1mm rain shows that more than 40000 observations are around 0.

plt.hist(df_traffic_data.rain_1h.loc[df_traffic_data.rain_1h<1])

plt.show()
#Plotting snow variable indicates that data is again skewed and most of the observations have value close to 0.0.

plt.hist(df_traffic_data.snow_1h)

plt.show()
#clouds_all indicates the cloud coverage for the give day and hour

sns.distplot(df_traffic_data.clouds_all)

plt.show()
#exploring different categories in weather_main

sns.countplot(y='weather_main', data=df_traffic_data)
#exploring different categories in weather_description

plt.figure(figsize=(10,8))

sns.countplot(y='weather_description', data=df_traffic_data)

plt.show()
#Exploring traffic volume on holidays

plt.figure(figsize=(10,8))

sns.boxplot(y='holiday',x='traffic_volume', data = holidays)

plt.show()
#Plotting relationship between temp, rain_1h, snow_1h, cloud_all.

num_vars = ['temp','rain_1h','snow_1h','clouds_all','traffic_volume']

from pandas.plotting import scatter_matrix

scatter_matrix(df_traffic_data[num_vars],figsize=(10,8))

plt.show()
#plotting temperature against traffic volume

plt.figure(figsize=(10,8))

sns.set_style('darkgrid')

sns.jointplot(y='traffic_volume', x='temp', data = df_traffic_data.loc[df_traffic_data.temp>-50])

plt.show()
#scatterplot between traffic_volume and temp

plt.figure(figsize=(8,6))

sns.scatterplot(y='traffic_volume', x='temp', data = df_traffic_data.loc[df_traffic_data.temp>-50])
#Plotting traffic volume over clouds_all

plt.figure(figsize=(14,8))

sns.barplot(x='clouds_all', y = 'traffic_volume', data = df_traffic_data)

plt.show()
#Plotting weather_main over traffic volume

plt.figure(figsize=(8,6))

sns.barplot(x='weather_main', y = 'traffic_volume', data = df_traffic_data)

plt.show()
#Plotting weather_description over traffic volume

plt.figure(figsize=(12,8))

sns.barplot(y='weather_description', x = 'traffic_volume', data = df_traffic_data)

plt.show()
#correlation between different numeric variables. plot shows no strong correlation between traffic and other variables

sns.heatmap(df_traffic_data.corr(), annot=True)

plt.show()
#copying data to new data frame

df_traffic_features = df_traffic_data.copy()
#Extracting features from date_time variable

df_traffic_features['date_time'] = pd.to_datetime(df_traffic_features.date_time)

df_traffic_features['weekday'] = df_traffic_features.date_time.dt.weekday

df_traffic_features['date'] = df_traffic_features.date_time.dt.date

df_traffic_features['hour'] = df_traffic_features.date_time.dt.hour

df_traffic_features['month'] = df_traffic_features.date_time.dt.month

df_traffic_features['year'] = df_traffic_features.date_time.dt.year

#Monday is 0 and Sunday is 6
df_traffic_features.head()
#categorizing hours to different time periods like morning, afternoon etc

def hour_modify(x):

    Early_Morning = [4,5,6,7]

    Morning = [8,9,10,11]

    Afternoon = [12,13,14,15]

    Evening = [16,17,18,19]

    Night = [20,21,22,23]

    Late_Night = [24,1,2,3]

    if x in Early_Morning:

        return 'Early_Morning'

    elif x in Morning:

        return 'Morning'

    elif x in Afternoon:

        return 'Afternoon'

    elif x in Evening:

        return 'Evening'

    elif x in Night:

        return 'Night'

    else:

        return 'Late_Night'

    

df_traffic_features['hour'] = df_traffic_features.hour.map(hour_modify)
#Traffic volume plotted against weekday. Weekends show less traffic volume.

plt.figure(figsize=(8,6))

sns.boxplot(x='weekday', y='traffic_volume', data = df_traffic_features)

plt.show()
#aggreagating traffic volume over year and plotting 



df_date_traffic = df_traffic_features.groupby('year').aggregate({'traffic_volume':'mean'})

plt.figure(figsize=(8,6))

sns.lineplot(x = df_date_traffic.index, y = df_date_traffic.traffic_volume, data = df_date_traffic)

plt.show()
#Other holidays are very sparse compared to none holidays. 

#Hence encoding the holidays as TRUE and none Holidays as FALSE



def modify_holiday(x):

    if x == 'None':

        return False

    else:

        return True

df_traffic_features['holiday'] = df_traffic_features['holiday'].map(modify_holiday)
#Outlier in temp which was detected earlier needs to be removed

df_traffic_features = df_traffic_features.loc[df_traffic_features.temp>-250]
#Traffic volume difference during holiday and non holiday

plt.figure(figsize=(8,6))

sns.barplot(x='holiday', y='traffic_volume', data = df_traffic_features)

plt.show()
#clouds, rain and snow distribution over different weather conditions

df_traffic_features.groupby('weather_description').aggregate({'traffic_volume':[np.mean,np.size],

                                                              'clouds_all':'count','rain_1h':'mean','snow_1h':'mean'})
df_traffic_features['weather_description'] = df_traffic_features['weather_description'].map(lambda x:x.lower())
#The weather description mostly describes rain, snow, thunderstorms, fog, mist and haze.



#I will create following new columns:

#thunderstorm - True where weather description contains Thunderstorm else False

#fog - True where weather description contains fog else False

#mist - True where weather description contains mist else False

#haze - True where weather description contains haze else False
#Any row containing "thunderstorm" is replaced by "thunderstorm"

df_traffic_features.loc[df_traffic_features['weather_description'].str.contains('thunderstorm'),'weather_description'] = 'thunderstorm'    
weather = ['thunderstorm','mist','fog','haze']

df_traffic_features.loc[np.logical_not(df_traffic_features['weather_description'].isin(weather)),'weather_description'] = 'other'
df_traffic_features.weather_description.value_counts()
#creating dummy variables for these newly created categories in weather description

df_traffic_features = pd.get_dummies(columns=['weather_description'],data=df_traffic_features)
df_traffic_features.rename(columns={'weather_description_fog':'fog', 'weather_description_haze':'haze',

                                   'weather_description_mist':'mist', 'weather_description_thunderstorm':'thunderstorm'}, inplace = True)

df_traffic_features.drop(columns = ['weather_description_other', 'weather_main'], inplace = True)
df_traffic_features.columns
#Plotiing rain data shows one outlier data point. Lets remove it.

plt.figure(figsize=(8,6))

sns.boxplot('rain_1h',data = df_traffic_features)

plt.show()
sns.boxplot('rain_1h',data = df_traffic_features.loc[df_traffic_features.rain_1h<2000])
#Removing outlier in rain column and converting numeric data to categories

#rain value equal to 0.0 as no_rain

#rain value greater than 0.0 is cut into 3 quantiles



df_traffic_features = df_traffic_features.loc[df_traffic_features.rain_1h<2000]

df_traffic_features_temp = df_traffic_features.loc[df_traffic_features.rain_1h>0]

rain_q = pd.DataFrame(pd.qcut(df_traffic_features_temp['rain_1h'] ,q=3, labels=['light','moderate','heavy']))

df_traffic_cat = df_traffic_features.merge(rain_q,left_index=True, right_index=True, how='left')

df_traffic_cat['rain_1h_y'] = df_traffic_cat.rain_1h_y.cat.add_categories('no_rain')

df_traffic_cat['rain_1h_y'].fillna('no_rain', inplace = True) #no_rain is not in the category, adding it and filling



df_traffic_cat.drop(columns=['rain_1h_x'], inplace = True)

df_traffic_cat.rename(columns={'rain_1h_y':'rain_1h'}, inplace = True)

df_traffic_cat.head()
#Plotiing snow data shows that it is extremely skewed as observed during univariate analysis

sns.boxplot('snow_1h',data = df_traffic_features)
#only 63 observations have snow greater than 0.0, it can be encoded as no_snow and 

df_traffic_features.snow_1h[df_traffic_features.snow_1h>0].count()

#63 columns -> change to snow, no_snow
def modify_snow1h(x):

    if x==0:

        return 'no_snow'

    else:

        return 'snow'

    

df_date_traffic['snow_1h'] = df_traffic_cat.snow_1h.map(modify_snow1h)
df_traffic_features.head()
#setting date as index

df_traffic_cat.set_index('date', inplace = True)
df_traffic_cat.columns
target = ['traffic_volume']

cat_vars = ['holiday', 'snow_1h','weekday', 'hour', 'month', 'year', 'fog', 'haze','mist', 'thunderstorm', 'rain_1h']

num_vars = ['temp','clouds_all']
#Creating pipeline to transform data

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer



numeric_transformer = Pipeline(steps=[

    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[

    ('oneHot',OneHotEncoder())])



preprocessor = ColumnTransformer(transformers=[

    ('num',numeric_transformer,num_vars),

    ('cat',categorical_transformer,cat_vars)])



df_traffic_transformed = preprocessor.fit_transform(df_traffic_cat).toarray()
#Splitting data into train and test data



X_train = df_traffic_transformed[:32290]

X_test = df_traffic_transformed[32291:]

y_train = df_traffic_cat.traffic_volume[:32290]

y_test = df_traffic_cat.traffic_volume[32291:]
#Fitting XGBoost regressor and parameter tuning using Grid search

from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import GridSearchCV

import xgboost as xgb



tscv = TimeSeriesSplit(n_splits=3)

model = xgb.XGBRegressor()



param_grid = {'nthread':[4,6,8], 

              'objective':['reg:linear'],

              'learning_rate': [.03, 0.05, .07],

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500]}



GridSearch = GridSearchCV(estimator = model,param_grid= param_grid,cv=tscv, n_jobs = 2 )

GridSearch.fit(X_train, y_train)

y_pred = GridSearch.predict(X_test)
#Root mean square

from sklearn.metrics import mean_squared_error

RMSE = np.sqrt(mean_squared_error(y_pred=y_pred,y_true=y_test))

print(RMSE)
#RMSE is not so great. Next I will explore models specfic for time series data like ARIMA
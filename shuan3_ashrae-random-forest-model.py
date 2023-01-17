%matplotlib inline

import pandas as pd

import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection  import train_test_split

import numpy as np

from scipy.stats import norm # for scientific Computing

from scipy import stats, integrate

import matplotlib.pyplot as plt

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.




# Any results you write to the current directory are saved as output.

ASHRAE_train =  pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

ASHRAE_test=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

weather_train=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

weather_test=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

building_meta=pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
ASHRAE_train.info()

weather_train.info()

## Function to reduce the DF size

def reduce_memory_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

reduce_memory_usage(building_meta)

reduce_memory_usage(weather_train)

reduce_memory_usage(ASHRAE_train)



reduce_memory_usage(weather_test)

reduce_memory_usage(ASHRAE_test)

ASHRAE_train.describe()
print('Size of the building dataset is', building_meta.shape)

print('Size of the weather_train dataset is', weather_train.shape)

print('Size of the train dataset is', ASHRAE_train.shape)
ASHRAE_train.describe()
building_meta.describe()
primary_use_numbersOfUniqueValue = building_meta['primary_use'].nunique()

 

print('Number of unique values in column "primary_use" of the building_meta : ')

print(primary_use_numbersOfUniqueValue)

primary_use_element = building_meta['primary_use'].unique()

 

print('Unique element in column "primary_use" of the building_meta : ')

print(primary_use_element)
print('Columns of the building dataset is', building_meta.columns)

print('Columns of the weather_train dataset is', weather_train.columns)

print('Columns of the train dataset is', ASHRAE_train.columns)
fig, ax = plt.subplots(figsize=(15,7))

sns.heatmap(building_meta.isnull(), yticklabels=False,cmap='viridis')


print("Percentage of missing values in the building_meta dataset")

building_meta.isna().sum()/len(building_meta)*100


print("Percentage of missing values in the train dataset")

ASHRAE_train.isna().sum()/len(ASHRAE_train)*100


print("Percentage of missing values in the weather_train dataset")

weather_train.isna().sum()/len(weather_train)*100
#pd.merge(df1, df2, on='employee')

BuildingTrainMerge=building_meta.merge(ASHRAE_train,left_on='building_id',right_on='building_id',how='left')

BuildingTrainMerge.shape
BTW_train=BuildingTrainMerge.merge(weather_train,left_on=['site_id','timestamp'],right_on=['site_id','timestamp'],how='left')

BTW_train.shape
BTW_train.columns
print("Percentage of missing values in the BTW_train dataset")

BTW_train.isna().sum()/len(BTW_train)*100
BTW_train.hist('sea_level_pressure')

BTW_train[['sea_level_pressure']].describe()
BTW_train.hist('cloud_coverage')

BTW_train[['cloud_coverage']].describe()
BTW_train.hist('precip_depth_1_hr')

BTW_train[['precip_depth_1_hr']].describe()
BTW_train.hist('wind_speed')

BTW_train[['wind_speed']].describe()
BTW_train.hist(column='air_temperature')

BTW_train[['air_temperature']].describe()
sns.boxplot(x = 'meter', y = 'meter_reading', data = BTW_train)
def outlier_function(df, col_name):

    ''' this function detects first and third quartile and interquartile range for a given column of a dataframe

    then calculates upper and lower limits to determine outliers conservatively

    returns the number of lower and uper limit and number of outliers respectively

    '''

    first_quartile = np.percentile(

        np.array(df[col_name].tolist()), 25)

    third_quartile = np.percentile(

        np.array(df[col_name].tolist()), 75)

    IQR = third_quartile - first_quartile

                      

    upper_limit = third_quartile+(3*IQR)

    lower_limit = first_quartile-(3*IQR)

    outlier_count = 0

                      

    for value in df[col_name].tolist():

        if (value < lower_limit) | (value > upper_limit):

            outlier_count +=1

    return lower_limit, upper_limit, outlier_count
print("{} percent of {} are outliers."

      .format((

              (100 * outlier_function(BTW_train, 'meter_reading')[2])

               / len(BTW_train['meter_reading'])),

              'meter_reading'))
# Distribution of the meter reading in meters without zeros

plt.figure(figsize=(12,10))



#list of different meters

meters = sorted(BTW_train['meter'].unique().tolist())



# plot meter_reading distribution for each meter

for meter_type in meters:

    subset = BTW_train[BTW_train['meter'] == meter_type]

    sns.kdeplot(np.log1p(subset["meter_reading"]), 

                label=meter_type, linewidth=2)



# set title, legends and labels

plt.ylabel("Density")

plt.xlabel("Meter_reading")

plt.legend(['electricity', 'chilled water', 'steam', 'hot water'])

plt.title("Density of Logartihm(Meter Reading + 1) Among Different Meters", size=14)
BTW_train.columns
corrmat=BTW_train.corr()

fig,ax=plt.subplots(figsize=(12,10))

sns.heatmap(corrmat,annot=True,annot_kws={'size': 12})

BTW_train = BTW_train.drop(columns=['year_built', 'floor_count', 'wind_direction', 'dew_temperature'])

BTW_train ['timestamp'] =  pd.to_datetime(BTW_train['timestamp'])

BTW_train['Month']=pd.DatetimeIndex(BTW_train['timestamp']).month

BTW_train['Day']=pd.DatetimeIndex(BTW_train['timestamp']).day

BTW_train= BTW_train.groupby(['meter',BTW_train['building_id'],'primary_use',BTW_train['Month'], BTW_train['Day']]).agg({'meter_reading':'sum', 'air_temperature': 'mean', 'wind_speed': 'mean', 'precip_depth_1_hr': 'mean', 'cloud_coverage': 'mean', 'square_feet': 'mean'})
BTW_train.columns
BTW_train = BTW_train.reset_index()
BTW_train.describe()
BTW_train['wind_speed'] = BTW_train['wind_speed'].astype('float32')

BTW_train['air_temperature'] = BTW_train['air_temperature'].astype('float32')

BTW_train['precip_depth_1_hr'] = BTW_train['precip_depth_1_hr'].astype('float32')

BTW_train['cloud_coverage'] = BTW_train['cloud_coverage'].astype('float32')
BTW_train['precip_depth_1_hr'].fillna(method='ffill', inplace = True)

BTW_train['cloud_coverage'].fillna(method='bfill', inplace = True)



BTW_train['wind_speed'].fillna(BTW_train['wind_speed'].mean(), inplace=True)

BTW_train['air_temperature'].fillna(BTW_train['air_temperature'].mean(), inplace=True)

BTW_train.isnull().sum()
BTW_train.shape
BTW_train.dtypes
BTW_train.columns


BTW_linearR = pd.get_dummies(BTW_train, columns=['primary_use'])
BTW_linearR.columns
X =BTW_linearR[['building_id', 'meter', 'air_temperature', 'wind_speed', 'precip_depth_1_hr', 'cloud_coverage',

       'square_feet', 'primary_use_Education', 'primary_use_Entertainment/public assembly',

       'primary_use_Food sales and service', 'primary_use_Healthcare',

       'primary_use_Lodging/residential',

       'primary_use_Manufacturing/industrial', 'primary_use_Office',

       'primary_use_Other', 'primary_use_Parking',

       'primary_use_Public services', 'primary_use_Religious worship',

       'primary_use_Retail', 'primary_use_Services',

       'primary_use_Technology/science', 'primary_use_Utility',

       'primary_use_Warehouse/storage', 'Month', 'Day']]



# Create target variable

y = BTW_linearR['meter_reading']



# Train, test, split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 0)
# Fit

# Import model

from sklearn.linear_model import LinearRegression



# Create linear regression object

regressor = LinearRegression()



# Fit model to training data

regressor.fit(X_train,y_train)
# Predicting test set results

y_pred = regressor.predict(X_test)
print('Accuracy %d', regressor.score(X_test, y_test))
#Calculate R Sqaured

print('R^2 =',metrics.explained_variance_score(y_test,y_pred))
cdf = pd.DataFrame(data = regressor.coef_, index = X.columns, columns = ['Coefficients'])

cdf
cdf.Coefficients.nlargest(10).plot(kind='barh')
import statsmodels.api as sm

from scipy import stats

X =BTW_linearR[['building_id', 'meter', 'air_temperature', 'wind_speed', 'precip_depth_1_hr', 'cloud_coverage',

       'square_feet', 'primary_use_Education', 'primary_use_Entertainment/public assembly',

       'primary_use_Food sales and service', 'primary_use_Healthcare',

       'primary_use_Lodging/residential',

       'primary_use_Manufacturing/industrial', 'primary_use_Office',

       'primary_use_Other', 'primary_use_Parking',

       'primary_use_Public services', 'primary_use_Religious worship',

       'primary_use_Retail', 'primary_use_Services',

       'primary_use_Technology/science', 'primary_use_Utility',

       'primary_use_Warehouse/storage', 'Month', 'Day']]



# Create target variable

y = BTW_linearR['meter_reading']

 

 

 

X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
K =BTW_linearR[['meter','wind_speed', 'cloud_coverage',

                'primary_use_Education','primary_use_Entertainment/public assembly', 'primary_use_Healthcare',

       'primary_use_Manufacturing/industrial', 'primary_use_Office',

       'primary_use_Other', 'primary_use_Parking','primary_use_Religious worship',

       'primary_use_Retail','primary_use_Technology/science', 'primary_use_Utility', 'Month']]



# Create target variable

y = BTW_linearR['meter_reading']
lm = LinearRegression()



# Fit model to training data

lm.fit(K,y)
# Train, test, split

from sklearn.model_selection import train_test_split

K_train, K_test, y_train, y_test = train_test_split(K,y, test_size = .20, random_state= 0)
print('Accuracy %d', lm.score(K_test, y_test))
y_pred = lm.predict(K_test)
print('R^2 =',metrics.explained_variance_score(y_test,y_pred))
lm.score(K,y)
regressor.score(X_train,y_train)
cdf1 = pd.DataFrame(data = lm.coef_, index = K.columns, columns = ['Coefficients'])
cdf1 .Coefficients.nlargest(10).plot(kind='barh')
XD =BTW_linearR[['building_id', 'meter', 'air_temperature', 'wind_speed', 'precip_depth_1_hr', 'cloud_coverage',

       'square_feet', 'primary_use_Education', 'primary_use_Entertainment/public assembly',

       'primary_use_Food sales and service', 'primary_use_Healthcare',

       'primary_use_Lodging/residential',

       'primary_use_Manufacturing/industrial', 'primary_use_Office',

       'primary_use_Other', 'primary_use_Parking',

       'primary_use_Public services', 'primary_use_Religious worship',

       'primary_use_Retail', 'primary_use_Services',

       'primary_use_Technology/science', 'primary_use_Utility',

       'primary_use_Warehouse/storage', 'Month', 'Day']]



# Create target variable

YD = BTW_linearR['meter_reading']



# Train, test, split

from sklearn.model_selection import train_test_split

XD_train,XD_test, YD_train, YD_test = train_test_split(XD,YD, test_size = .20, random_state= 0)
from sklearn.tree import DecisionTreeRegressor

regr_depth2 = DecisionTreeRegressor(max_depth=2)

regr_depth5 = DecisionTreeRegressor(max_depth=5)

regr_depth2.fit(XD_train, YD_train)

regr_depth5.fit(XD_train, YD_train)
y_1 = regr_depth2.predict(XD_test)

y_2 = regr_depth5.predict(XD_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_1})

df.head()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_1))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_1))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_1)))
#Calculate R Sqaured

print('R^2 =',metrics.explained_variance_score(y_test,y_1))
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_2})

df.head()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_2))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_2))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_2)))
print('R^2 =',metrics.explained_variance_score(y_test,y_2))
plt.plot(XD_test, y_1, color="blue",label="max_depth=2", linewidth=2)

plt.plot(XD_test, y_2, color="green", label="max_depth=5", linewidth=2)

plt.xlabel("data")

plt.ylabel("target")

plt.title("Decision Tree Regression")

plt.show()
print('Accuracy %d', regr_depth2.score(XD_train, YD_train))
print('Accuracy %d', regr_depth5.score(XD_train, YD_train))
yd_pred = regr_depth5.predict(XD_test)
yd_pred
print('XD 19',XD.columns[19], 'XD 6',XD.columns[6],'X0',XD.columns[0],'X1',XD.columns[1],'X23',XD.columns[23],'X19',XD.columns[19],'X2',XD.columns[2])
YD.describe()
XD.columns
feat_importancesDT = pd.Series(regr_depth5.feature_importances_, index=XD.columns)

feat_importancesDT.nlargest(10).plot(kind='barh')
import sklearn.ensemble as ske

import matplotlib.pyplot as plt

RFR = ske.RandomForestRegressor()

RFR.fit(XD,YD)
RFR.score(XD,YD)
YR_pred = RFR.predict(XD_test)
feat_importancesRFR = pd.Series(RFR.feature_importances_, index=XD.columns)

feat_importancesRFR.nlargest(10).plot(kind='barh')
#pip install pydot
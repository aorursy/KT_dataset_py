# loading the required libraries 

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

from fancyimpute import KNN

import warnings

warnings.filterwarnings('ignore')

from geopy.distance import geodesic

from geopy.distance import great_circle

from scipy.stats import chi2_contingency

import statsmodels.api as sm

from statsmodels.formula.api import ols

from patsy import dmatrices

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import metrics

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

import xgboost as xgb

from sklearn.externals import joblib 
# set the working directory

os.chdir('C:/Users/admin/Documents/Python Files')

os.getcwd()
# Importing data

train = pd.read_csv('train_cab.csv',dtype={'fare_amount':np.float64},na_values={'fare_amount':'430-'})

test = pd.read_csv('test.csv')

data=[train,test]

for i in data:

    i['pickup_datetime']  = pd.to_datetime(i['pickup_datetime'],errors='coerce')

train.head(5)
train.info()
test.head(5)
test.info()
test.describe()
train.describe()
cat_var=['passenger_count']

num_var=['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
# setting up the sns for plots

sns.set(style='darkgrid',palette='Set1')
plt.figure(figsize=(20,20))

plt.subplot(321)

_ = sns.distplot(train['fare_amount'],bins=50)

plt.subplot(322)

_ = sns.distplot(train['pickup_longitude'],bins=50)

plt.subplot(323)

_ = sns.distplot(train['pickup_latitude'],bins=50)

plt.subplot(324)

_ = sns.distplot(train['dropoff_longitude'],bins=50)

plt.subplot(325)

_ = sns.distplot(train['dropoff_latitude'],bins=50)

# plt.savefig('hist.png')

plt.show()
# plt.figure(figsize=(25,25))

# _ = sns.swarmplot(x='passenger_count',y='fare_amount',data=train)

# plt.title('Cab Fare w.r.t passenger_count')
_ = sns.jointplot(x='fare_amount',y='pickup_longitude',data=train,kind = 'reg')

_.annotate(stats.pearsonr)

# plt.savefig('jointfplo.png')

plt.show()
_ = sns.jointplot(x='fare_amount',y='pickup_latitude',data=train,kind = 'reg')

_.annotate(stats.pearsonr)

# plt.savefig('jointfpla.png')

plt.show()
_ = sns.jointplot(x='fare_amount',y='dropoff_longitude',data=train,kind = 'reg')

_.annotate(stats.pearsonr)

# plt.savefig('jointfdlo.png')

plt.show()
_ = sns.jointplot(x='fare_amount',y='dropoff_latitude',data=train,kind = 'reg')

_.annotate(stats.pearsonr)

# plt.savefig('jointfdla.png')

plt.show()
plt.figure(figsize=(20,20))

plt.subplot(321)

_ = sns.violinplot(y='fare_amount',data=train)

plt.subplot(322)

_ = sns.violinplot(y='pickup_longitude',data=train)

plt.subplot(323)

_ = sns.violinplot(y='pickup_latitude',data=train)

plt.subplot(324)

_ = sns.violinplot(y='dropoff_longitude',data=train)

plt.subplot(325)

_ = sns.violinplot(y='dropoff_latitude',data=train)

plt.savefig('violin.png')

plt.show()
_ =sns.pairplot(data=train[num_var],kind='scatter',dropna=True)

_.fig.suptitle('Pairwise plot of all numerical variables')

# plt.savefig('Pairwise.png')

plt.show()
sum(train['fare_amount']<1)
train[train['fare_amount']<1]
train = train.drop(train[train['fare_amount']<1].index, axis=0)
# train.loc[train['fare_amount'] < 1,'fare_amount'] = np.nan
for i in range(4,11):

    print('passenger_count above' +str(i)+'={}'.format(sum(train['passenger_count']>i)))
train[train['passenger_count']>6]
train[train['passenger_count']<1]
len(train[train['passenger_count']<1])
test['passenger_count'].unique()
train = train.drop(train[train['passenger_count']>6].index, axis=0)

train = train.drop(train[train['passenger_count']<1].index, axis=0)
# train.loc[train['passenger_count'] >6,'passenger_count'] = np.nan

# train.loc[train['passenger_count'] >1,'passenger_count'] = np.nan
sum(train['passenger_count']>6)
print('pickup_longitude above 180={}'.format(sum(train['pickup_longitude']>180)))

print('pickup_longitude below -180={}'.format(sum(train['pickup_longitude']<-180)))

print('pickup_latitude above 90={}'.format(sum(train['pickup_latitude']>90)))

print('pickup_latitude below -90={}'.format(sum(train['pickup_latitude']<-90)))

print('dropoff_longitude above 180={}'.format(sum(train['dropoff_longitude']>180)))

print('dropoff_longitude below -180={}'.format(sum(train['dropoff_longitude']<-180)))

print('dropoff_latitude below -90={}'.format(sum(train['dropoff_latitude']<-90)))

print('dropoff_latitude above 90={}'.format(sum(train['dropoff_latitude']>90)))
for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:

    print(i,'equal to 0={}'.format(sum(train[i]==0)))
train = train.drop(train[train['pickup_latitude']>90].index, axis=0)

for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:

    train = train.drop(train[train[i]==0].index, axis=0)
# for i in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:

#     train.loc[train[i]==0,i] = np.nan

# train.loc[train['pickup_latitude']>90,'pickup_latitude'] = np.nan
train.shape
df=train.copy()

# train=df.copy()
#Create dataframe with missing percentage

missing_val = pd.DataFrame(train.isnull().sum())

#Reset index

missing_val = missing_val.reset_index()

missing_val
#Rename variable

missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

missing_val

#Calculate percentage

missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(train))*100

#descending order

missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

missing_val
# Choosing a random values to replace it as NA

train['passenger_count'].loc[1000]
# Replacing 1.0 with NA

train['passenger_count'].loc[1000] = np.nan

train['passenger_count'].loc[1000]
# Impute with mode

train['passenger_count'].fillna(train['passenger_count'].mode()[0]).loc[1000]
# for i in ['fare_amount','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:

#     # Choosing a random values to replace it as NA

#     a=train[i].loc[1000]

#     print(i,'at loc-1000:{}'.format(a))

#     # Replacing 1.0 with NA

#     train[i].loc[1000] = np.nan

#     print('Value after replacing with nan:{}'.format(train[i].loc[1000]))

#     # Impute with mean

#     print('Value if imputed with mean:{}'.format(train[i].fillna(train[i].mean()).loc[1000]))

#     # Impute with median

#     print('Value if imputed with median:{}\n'.format(train[i].fillna(train[i].median()).loc[1000]))
# Choosing a random values to replace it as NA

a=train['fare_amount'].loc[1000]

print('fare_amount at loc-1000:{}'.format(a))

# Replacing 1.0 with NA

train['fare_amount'].loc[1000] = np.nan

print('Value after replacing with nan:{}'.format(train['fare_amount'].loc[1000]))

# Impute with mean

print('Value if imputed with mean:{}'.format(train['fare_amount'].fillna(train['fare_amount'].mean()).loc[1000]))

# Impute with median

print('Value if imputed with median:{}'.format(train['fare_amount'].fillna(train['fare_amount'].median()).loc[1000]))
train.std()
columns=['fare_amount', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude', 'passenger_count']
pickup_datetime=pd.DataFrame(train['pickup_datetime'])
# Imputing with missing values using KNN

# Use 19 nearest rows which have a feature to fill in each row's missing features

train = pd.DataFrame(KNN(k = 19).fit_transform(train.drop('pickup_datetime',axis=1)),columns=columns, index=train.index)
train.std()
train.loc[1000]
train['passenger_count'].head()
train['passenger_count']=train['passenger_count'].astype('int')
train.std()
train['passenger_count'].unique()
train['passenger_count']=train['passenger_count'].round().astype('object').astype('category',ordered=True)
train['passenger_count'].unique()
train.loc[1000]
pickup_datetime.head()
#Create dataframe with missing percentage

missing_val = pd.DataFrame(pickup_datetime.isnull().sum())

#Reset index

missing_val = missing_val.reset_index()

missing_val
pickup_datetime.shape
train.shape
# df1 = train.copy()

train=df1.copy()
train['passenger_count'].describe()
train.describe()
plt.figure(figsize=(20,5)) 

plt.xlim(0,100)

sns.boxplot(x=train['fare_amount'],data=train,orient='h')

plt.title('Boxplot of fare_amount')

# plt.savefig('bp of fare_amount.png')

plt.show()
# sum(train['fare_amount']<22.5)/len(train['fare_amount'])*100
plt.figure(figsize=(20,10))

plt.xlim(0,100)

_ = sns.boxplot(x=train['fare_amount'],y=train['passenger_count'],data=train,orient='h')

plt.title('Boxplot of fare_amount w.r.t passenger_count')

# plt.savefig('Boxplot of fare_amount w.r.t passenger_count.png')

plt.show()
train.describe()
train['passenger_count'].describe()
def outlier_treatment(col):

    ''' calculating outlier indices and replacing them with NA  '''

    #Extract quartiles

    q75, q25 = np.percentile(train[col], [75 ,25])

    print(q75,q25)

    #Calculate IQR

    iqr = q75 - q25

    #Calculate inner and outer fence

    minimum = q25 - (iqr*1.5)

    maximum = q75 + (iqr*1.5)

    print(minimum,maximum)

    #Replace with NA

    train.loc[train[col] < minimum,col] = np.nan

    train.loc[train[col] > maximum,col] = np.nan
# for i in num_var:

    outlier_treatment('fare_amount')

#     outlier_treatment('pickup_longitude')

#     outlier_treatment('pickup_latitude')

#     outlier_treatment('dropoff_longitude')

#     outlier_treatment('dropoff_latitude')
pd.DataFrame(train.isnull().sum())
train.std()
#Imputing with missing values using KNN

train = pd.DataFrame(KNN(k = 3).fit_transform(train), columns = train.columns, index=train.index)
train.std()
train['passenger_count'].describe()
train['passenger_count']=train['passenger_count'].astype('int').round().astype('object').astype('category')
train.describe()
train.head()
df2 = train.copy()

# train=df2.copy()
train.shape
# we will Join 2 Dataframes pickup_datetime and train

train = pd.merge(pickup_datetime,train,right_index=True,left_index=True)

train.head()
train.shape
train=train.reset_index(drop=True)
pd.DataFrame(train.isna().sum())
train=train.dropna()
data = [train,test]

for i in data:

    i["year"] = i["pickup_datetime"].apply(lambda row: row.year)

    i["month"] = i["pickup_datetime"].apply(lambda row: row.month)

#     i["day_of_month"] = i["pickup_datetime"].apply(lambda row: row.day)

    i["day_of_week"] = i["pickup_datetime"].apply(lambda row: row.dayofweek)

    i["hour"] = i["pickup_datetime"].apply(lambda row: row.hour)
# train_nodummies=train.copy()

# train=train_nodummies.copy()
plt.figure(figsize=(20,10))

sns.countplot(train['year'])

# plt.savefig('year.png')



plt.figure(figsize=(20,10))

sns.countplot(train['month'])

# plt.savefig('month.png')



plt.figure(figsize=(20,10))

sns.countplot(train['day_of_week'])

# plt.savefig('day_of_week.png')



plt.figure(figsize=(20,10))

sns.countplot(train['hour'])

# plt.savefig('hour.png')
def f(x):

    ''' for sessions in a day using hour column '''

    if (x >=5) and (x <= 11):

        return 'morning'

    elif (x >=12) and (x <=16 ):

        return 'afternoon'

    elif (x >= 17) and (x <= 20):

        return'evening'

    elif (x >=21) and (x <= 23) :

        return 'night_PM'

    elif (x >=0) and (x <=4):

        return'night_AM'
def g(x):

    ''' for seasons in a year using month column'''

    if (x >=3) and (x <= 5):

        return 'spring'

    elif (x >=6) and (x <=8 ):

        return 'summer'

    elif (x >= 9) and (x <= 11):

        return'fall'

    elif (x >=12)|(x <= 2) :

        return 'winter'
def h(x):

    ''' for week:weekday/weekend in a day_of_week column '''

    if (x >=0) and (x <= 4):

        return 'weekday'

    elif (x >=5) and (x <=6 ):

        return 'weekend'
train['session'] = train['hour'].apply(f)

test['session'] = test['hour'].apply(f)

# train_nodummies['session'] = train_nodummies['hour'].apply(f)
train['seasons'] = train['month'].apply(g)

test['seasons'] = test['month'].apply(g)

# train['seasons'] = test['month'].apply(g)
train['week'] = train['day_of_week'].apply(h)

test['week'] = test['day_of_week'].apply(h)
train.shape
test.shape
train['passenger_count'].describe()
#Creating dummies for each variable in passenger_count and merging dummies dataframe to both train and test dataframe

temp = pd.get_dummies(train['passenger_count'], prefix = 'passenger_count')

train = train.join(temp)

temp = pd.get_dummies(test['passenger_count'], prefix = 'passenger_count')

test = test.join(temp)

temp = pd.get_dummies(train['seasons'], prefix = 'season')

train = train.join(temp)

temp = pd.get_dummies(test['seasons'], prefix = 'season')

test = test.join(temp)

temp = pd.get_dummies(train['week'], prefix = 'week')

train = train.join(temp)

temp = pd.get_dummies(test['week'], prefix = 'week')

test = test.join(temp)

temp = pd.get_dummies(train['session'], prefix = 'session')

train = train.join(temp)

temp = pd.get_dummies(test['session'], prefix = 'session')

test = test.join(temp)

temp = pd.get_dummies(train['year'], prefix = 'year')

train = train.join(temp)

temp = pd.get_dummies(test['year'], prefix = 'year')

test = test.join(temp)
train.head()
test.head()
train.columns
train=train.drop(['passenger_count_1','season_fall','week_weekday','session_afternoon','year_2009'],axis=1)

test=test.drop(['passenger_count_1','season_fall','week_weekday','session_afternoon','year_2009'],axis=1)
# train.sort_values('pickup_datetime')
# def haversine(coord1, coord2):

#     '''Calculate distance the cab travelled from pickup and dropoff location using the Haversine Formula'''

#     data = [train, test]

#     for i in data:

#         lon1, lat1 = coord1

#         lon2, lat2 = coord2

#         R = 6371000  # radius of Earth in meters

#         phi_1 = np.radians(i[lat1])

#         phi_2 = np.radians(i[lat2])

#         delta_phi = np.radians(i[lat2] - i[lat1])

#         delta_lambda = np.radians(i[lon2] - i[lon1])

#         a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2

#         c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

#         meters = R * c  # output distance in meters

#         km = meters / 1000.0  # output distance in kilometers

#         miles = round(km, 3)/1.609344

#         i['distance'] = miles

# #     print(f"Distance: {miles} miles")

# #     return miles
# haversine(['pickup_longitude','pickup_latitude'],['dropoff_longitude','dropoff_latitude'])
# Calculate distance the cab travelled from pickup and dropoff location using great_circle from geopy library

data = [train, test]

for i in data:

    i['great_circle']=i.apply(lambda x: great_circle((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)

    i['geodesic']=i.apply(lambda x: geodesic((x['pickup_latitude'],x['pickup_longitude']), (x['dropoff_latitude'],   x['dropoff_longitude'])).miles, axis=1)
train.head()
test.head()
pd.DataFrame(train.isna().sum())
pd.DataFrame(test.isna().sum())
# train_nodummies=train_nodummies.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',

#        'dropoff_longitude', 'dropoff_latitude','great_circle'],axis = 1)

# test_nodummies=test.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',

#        'dropoff_longitude', 'dropoff_latitude','passenger_count_1', 'passenger_count_2', 'passenger_count_3',

#        'passenger_count_4', 'passenger_count_5', 'passenger_count_6',

#        'season_fall', 'season_spring', 'season_summer', 'season_winter',

#        'week_weekday', 'week_weekend', 'session_afternoon', 'session_evening',

#        'session_morning', 'session_night (AM)', 'session_night (PM)',

#        'year_2009', 'year_2010', 'year_2011', 'year_2012', 'year_2013',

#        'year_2014', 'year_2015', 'great_circle'],axis = 1)
train=train.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',

       'month', 'day_of_week', 'hour', 'session', 'seasons', 'week','great_circle'],axis=1)

test=test.drop(['pickup_datetime','pickup_longitude', 'pickup_latitude',

       'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'year',

       'month', 'day_of_week', 'hour', 'session', 'seasons', 'week','great_circle'],axis=1)
train.shape,test.shape
# test_nodummies.columns
# train_nodummies.columns
train.columns
test.columns
train.head()
test.head()
plt.figure(figsize=(20,5)) 

sns.boxplot(x=train['geodesic'],data=train,orient='h')

plt.title('Boxplot of geodesic ')

# plt.savefig('bp geodesic.png')

plt.show()
plt.figure(figsize=(20,5)) 

plt.xlim(0,100)

sns.boxplot(x=train['geodesic'],data=train,orient='h')

plt.title('Boxplot of geodesic ')

# plt.savefig('bp geodesic.png')

plt.show()
outlier_treatment('geodesic')
pd.DataFrame(train.isnull().sum())
#Imputing with missing values using KNN

train = pd.DataFrame(KNN(k = 3).fit_transform(train), columns = train.columns, index=train.index)
cat_var=['passenger_count_2',

       'passenger_count_3', 'passenger_count_4', 'passenger_count_5',

       'passenger_count_6', 'season_spring', 'season_summer',

       'season_winter', 'week_weekend',

       'session_evening', 'session_morning', 'session_night_AM',

       'session_night_PM', 'year_2010', 'year_2011',

       'year_2012', 'year_2013', 'year_2014', 'year_2015']

num_var=['fare_amount','geodesic']

train[cat_var]=train[cat_var].apply(lambda x: x.astype('category') )

test[cat_var]=test[cat_var].apply(lambda x: x.astype('category') ) 
# heatmap using correlation matrix

plt.figure(figsize=(15,15))

_ = sns.heatmap(train[num_var].corr(), square=True, cmap='RdYlGn',linewidths=0.5,linecolor='w',annot=True)

plt.title('Correlation matrix ')

# plt.savefig('correlation.png')

plt.show()
_ = sns.jointplot(x='fare_amount',y='geodesic',data=train,kind = 'reg')

_.annotate(stats.pearsonr)

# plt.savefig('jointct.png')

plt.show()
#loop for chi square values

for i in cat_var:

    for j in cat_var:

        if(i != j):

            chi2, p, dof, ex = chi2_contingency(pd.crosstab(train[i], train[j]))

            if(p < 0.05):

                print(i,"and",j,"are dependent on each other with",p,'----Remove')

            else:

                print(i,"and",j,"are independent on each other with",p,'----Keep')
train.columns
#ANOVA _1)+C(passenger_count_2)+C(passenger_count_3)+C(passenger_count_4)+C(passenger_count_5)+C(passenger_count_6)

model = ols('fare_amount ~ C(passenger_count_2)+C(passenger_count_3)+C(passenger_count_4)+C(passenger_count_5)+C(passenger_count_6)+C(season_spring)+C(season_summer)+C(season_winter)+C(week_weekend)+C(session_night_AM)+C(session_night_PM)+C(session_evening)+C(session_morning)+C(year_2010)+C(year_2011)+C(year_2012)+C(year_2013)+C(year_2014)+C(year_2015)',data=train).fit()

                

aov_table = sm.stats.anova_lm(model)

aov_table
# _1+passenger_count_2+passenger_count_3+passenger_count_4+passenger_count_5+passenger_count_6

outcome, predictors = dmatrices('fare_amount ~ geodesic+passenger_count_2+passenger_count_3+passenger_count_4+passenger_count_5+passenger_count_6+season_spring+season_summer+season_winter+week_weekend+session_night_AM+session_night_PM+session_evening+session_morning+year_2010+year_2011+year_2012+year_2013+year_2014+year_2015',train, return_type='dataframe')

# calculating VIF for each individual Predictors

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

vif["features"] = predictors.columns

vif
train[num_var].var()
sns.distplot(train['geodesic'],bins=50)

# plt.savefig('distplot.png')
plt.figure()

stats.probplot(train['geodesic'], dist='norm', fit=True,plot=plt)

# plt.savefig('qq prob plot.png')
#Normalization

train['geodesic'] = (train['geodesic'] - min(train['geodesic']))/(max(train['geodesic']) - min(train['geodesic']))

test['geodesic'] = (test['geodesic'] - min(test['geodesic']))/(max(test['geodesic']) - min(test['geodesic']))
train['geodesic'].var()
sns.distplot(train['geodesic'],bins=50)

plt.savefig('distplot.png')
plt.figure()

stats.probplot(train['geodesic'], dist='norm', fit=True,plot=plt)

# plt.savefig('qq prob plot.png')
train.columns
# df4=train.copy()

train=df4.copy()

# f4=test.copy()

test=f4.copy()
train=train.drop(['passenger_count_2'],axis=1)

test=test.drop(['passenger_count_2'],axis=1)
train.columns
X = train.drop('fare_amount',axis=1).values

y = train['fare_amount'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

print(train.shape, X_train.shape, X_test.shape,y_train.shape,y_test.shape)
def rmsle(y,y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

def scores(y, y_):

    print('r square  ', metrics.r2_score(y, y_))

    print('Adjusted r square:{}'.format(1 - (1-metrics.r2_score(y, y_))*(len(y)-1)/(len(y)-X_train.shape[1]-1)))

    print('MAPE:{}'.format(np.mean(np.abs((y - y_) / y))*100))

    print('MSE:', metrics.mean_squared_error(y, y_))

    print('RMSE:', np.sqrt(metrics.mean_squared_error(y, y_))) 

def test_scores(model):

    print('<<<------------------- Training Data Score --------------------->')

    print()

    #Predicting result on Training data

    y_pred = model.predict(X_train)

    scores(y_train,y_pred)

    print('RMSLE:',rmsle(y_train,y_pred))

    print()

    print('<<<------------------- Test Data Score --------------------->')

    print()

    # Evaluating on Test Set

    y_pred = model.predict(X_test)

    scores(y_test,y_pred)

    print('RMSLE:',rmsle(y_test,y_pred))
# Setup the parameters and distributions to sample from: param_dist

param_dist = {'copy_X':[True, False],

          'fit_intercept':[True,False]}

# Instantiate a Decision reg classifier: reg

reg = LinearRegression()



# Instantiate the gridSearchCV object: reg_cv

reg_cv = GridSearchCV(reg, param_dist, cv=5,scoring='r2')



# Fit it to the data

reg_cv.fit(X, y)



# Print the tuned parameters and score

print("Tuned Decision reg Parameters: {}".format(reg_cv.best_params_))

print("Best score is {}".format(reg_cv.best_score_))
# Create the regressor: reg_all

reg_all = LinearRegression(copy_X= True, fit_intercept=True)



# Fit the regressor to the training data

reg_all.fit(X_train,y_train)



# Predict on the test data: y_pred

y_pred = reg_all.predict(X_test)



# Compute and print R^2 and RMSE

print("R^2: {}".format(reg_all.score(X_test, y_test)))

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print("Root Mean Squared Error: {}".format(rmse))

test_scores(reg_all)



# Compute and print the coefficients

reg_coef = reg_all.coef_

print(reg_coef)



# Plot the coefficients

plt.figure(figsize=(15,5))

plt.plot(range(len(test.columns)), reg_coef)

plt.xticks(range(len(test.columns)), test.columns.values, rotation=60)

plt.margins(0.02)

plt.savefig('linear coefficients')

plt.show()
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg

reg = LinearRegression()



# Compute 5-fold cross-validation scores: cv_scores

cv_scores = cross_val_score(reg,X,y,cv=5,scoring='neg_mean_squared_error')



# Print the 5-fold cross-validation scores

print(cv_scores)



print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Setup the parameters and distributions to sample from: param_dist

param_dist = {'alpha':np.logspace(-4, 0, 50),

          'normalize':[True,False],

             'max_iter':range(500,5000,500)}

# Instantiate a Decision ridge classifier: ridge

ridge = Ridge()



# Instantiate the gridSearchCV object: ridge_cv

ridge_cv = GridSearchCV(ridge, param_dist, cv=5,scoring='r2')



# Fit it to the data

ridge_cv.fit(X, y)



# Print the tuned parameters and score

print("Tuned Decision ridge Parameters: {}".format(ridge_cv.best_params_))

print("Best score is {}".format(ridge_cv.best_score_))
# Instantiate a ridge regressor: ridge

ridge = Ridge(alpha=0.0005428675439323859, normalize=True,max_iter = 500)



# Fit the regressor to the data

ridge.fit(X_train,y_train)



# Compute and print the coefficients

ridge_coef = ridge.coef_

print(ridge_coef)



# Plot the coefficients

plt.figure(figsize=(15,5))

plt.plot(range(len(test.columns)), ridge_coef)

plt.xticks(range(len(test.columns)), test.columns.values, rotation=60)

plt.margins(0.02)

# plt.savefig('ridge coefficients')

plt.show()

test_scores(ridge)
# Setup the parameters and distributions to sample from: param_dist

param_dist = {'alpha':np.logspace(-4, 0, 50),

          'normalize':[True,False],

             'max_iter':range(500,5000,500)}

# Instantiate a Decision lasso classifier: lasso

lasso = Lasso()



# Instantiate the gridSearchCV object: lasso_cv

lasso_cv = GridSearchCV(lasso, param_dist, cv=5,scoring='r2')



# Fit it to the data

lasso_cv.fit(X, y)



# Print the tuned parameters and score

print("Tuned Decision lasso Parameters: {}".format(lasso_cv.best_params_))

print("Best score is {}".format(lasso_cv.best_score_))
# Instantiate a lasso regressor: lasso

lasso = Lasso(alpha=0.00021209508879201905, normalize=False,max_iter = 500)



# Fit the regressor to the data

lasso.fit(X,y)



# Compute and print the coefficients

lasso_coef = lasso.coef_

print(lasso_coef)



# Plot the coefficients

plt.figure(figsize=(15,5))

plt.ylim(-1,10)

plt.plot(range(len(test.columns)), lasso_coef)

plt.xticks(range(len(test.columns)), test.columns.values, rotation=60)

plt.margins(0.02)

plt.savefig('lasso coefficients')

plt.show()

test_scores(lasso)
train.info()
# Setup the parameters and distributions to sample from: param_dist

param_dist = {'max_depth': range(2,16,2),

              'min_samples_split': range(2,16,2)}



# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeRegressor()



# Instantiate the gridSearchCV object: tree_cv

tree_cv = GridSearchCV(tree, param_dist, cv=5)



# Fit it to the data

tree_cv.fit(X, y)



# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))

print("Best score is {}".format(tree_cv.best_score_))
# Instantiate a tree regressor: tree

tree = DecisionTreeRegressor(max_depth= 6, min_samples_split=2)



# Fit the regressor to the data

tree.fit(X_train,y_train)



# Compute and print the coefficients

tree_features = tree.feature_importances_

print(tree_features)



# Sort test importances in descending order

indices = np.argsort(tree_features)[::1]



# Rearrange test names so they match the sorted test importances

names = [test.columns[i] for i in indices]



# Creating plot

fig = plt.figure(figsize=(20,10))

plt.title("test Importance")



# Add horizontal bars

plt.barh(range(pd.DataFrame(X_train).shape[1]),tree_features[indices],align = 'center')

plt.yticks(range(pd.DataFrame(X_train).shape[1]), names)

plt.savefig('tree test importance')

plt.show()

# Make predictions and cal error

test_scores(tree)
# Create the random grid

random_grid = {'n_estimators': range(100,500,100),

               'max_depth': range(5,20,1),

               'min_samples_leaf':range(2,5,1),

              'max_features':['auto','sqrt','log2'],

              'bootstrap': [True, False],

              'min_samples_split': range(2,5,1)}

# Instantiate a Decision Forest classifier: Forest

Forest = RandomForestRegressor()



# Instantiate the gridSearchCV object: Forest_cv

Forest_cv = RandomizedSearchCV(Forest, random_grid, cv=5)



# Fit it to the data

Forest_cv.fit(X, y)



# Print the tuned parameters and score

print("Tuned Random Forest Parameters: {}".format(Forest_cv.best_params_))

print("Best score is {}".format(Forest_cv.best_score_))
# Instantiate a Forest regressor: Forest

Forest = RandomForestRegressor(n_estimators=100, min_samples_split= 2, min_samples_leaf=4, max_features='auto', max_depth=9, bootstrap=True)



# Fit the regressor to the data

Forest.fit(X_train,y_train)



# Compute and print the coefficients

Forest_features = Forest.feature_importances_

print(Forest_features)



# Sort feature importances in descending order

indices = np.argsort(Forest_features)[::1]



# Rearrange feature names so they match the sorted feature importances

names = [test.columns[i] for i in indices]



# Creating plot

fig = plt.figure(figsize=(20,10))

plt.title("Feature Importance")



# Add horizontal bars

plt.barh(range(pd.DataFrame(X_train).shape[1]),Forest_features[indices],align = 'center')

plt.yticks(range(pd.DataFrame(X_train).shape[1]), names)

plt.savefig('Random forest feature importance')

plt.show()# Make predictions

test_scores(Forest)
from sklearn.model_selection import cross_val_score

# Create a random forest regression object: Forest

Forest = RandomForestRegressor(n_estimators=400, min_samples_split= 2, min_samples_leaf=4, max_features='auto', max_depth=12, bootstrap=True)



# Compute 5-fold cross-validation scores: cv_scores

cv_scores = cross_val_score(Forest,X,y,cv=5,scoring='neg_mean_squared_error')



# Print the 5-fold cross-validation scores

print(cv_scores)



print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
data_dmatrix = xgb.DMatrix(data=X,label=y)

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test)
dtrain,dtest,data_dmatrix
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}



cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()
# the final boosting round metric

print((cv_results["test-rmse-mean"]).tail(1))
Xgb = XGBRegressor()

Xgb.fit(X_train,y_train)

# pred_xgb = model_xgb.predict(X_test)

test_scores(Xgb)
# Create the random grid

para = {'n_estimators': range(100,500,100),

               'max_depth': range(3,10,1),

        'reg_alpha':np.logspace(-4, 0, 50),

        'subsample': np.arange(0.1,1,0.2),

        'colsample_bytree': np.arange(0.1,1,0.2),

        'colsample_bylevel': np.arange(0.1,1,0.2),

        'colsample_bynode': np.arange(0.1,1,0.2),

       'learning_rate': np.arange(.05, 1, .05)}

# Instantiate a Decision Forest classifier: Forest

Xgb = XGBRegressor()



# Instantiate the gridSearchCV object: Forest_cv

xgb_cv = RandomizedSearchCV(Xgb, para, cv=5)



# Fit it to the data

xgb_cv.fit(X, y)



# Print the tuned parameters and score

print("Tuned Xgboost Parameters: {}".format(xgb_cv.best_params_))

print("Best score is {}".format(xgb_cv.best_score_))

# Instantiate a xgb regressor: xgb

Xgb = XGBRegressor(subsample= 0.1, reg_alpha= 0.08685113737513521, n_estimators= 200, max_depth= 3, learning_rate=0.05, colsample_bytree= 0.7000000000000001, colsample_bynode=0.7000000000000001, colsample_bylevel=0.9000000000000001)



# Fit the regressor to the data

Xgb.fit(X_train,y_train)



# Compute and print the coefficients

xgb_features = Xgb.feature_importances_

print(xgb_features)



# Sort feature importances in descending order

indices = np.argsort(xgb_features)[::1]



# Rearrange feature names so they match the sorted feature importances

names = [test.columns[i] for i in indices]



# Creating plot

fig = plt.figure(figsize=(20,10))

plt.title("Feature Importance")



# Add horizontal bars

plt.barh(range(pd.DataFrame(X_train).shape[1]),xgb_features[indices],align = 'center')

plt.yticks(range(pd.DataFrame(X_train).shape[1]), names)

plt.savefig(' xgb feature importance')

plt.show()# Make predictions

test_scores(Xgb)
def rmsle(y,y_):

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))

def score(y, y_):

    print('r square  ', metrics.r2_score(y, y_))

    print('Adjusted r square:{}'.format(1 - (1-metrics.r2_score(y, y_))*(len(y)-1)/(len(y)-X_train.shape[1]-1)))

    print('MAPE:{}'.format(np.mean(np.abs((y - y_) / y))*100))

    print('MSE:', metrics.mean_squared_error(y, y_))

    print('RMSE:', np.sqrt(metrics.mean_squared_error(y, y_)))

    print('RMSLE:',rmsle(y_test,y_pred))

def scores(model):

    print('<<<------------------- Training Data Score --------------------->')

    print()

    #Predicting result on Training data

    y_pred = model.predict(X)

    score(y,y_pred)

    print('RMSLE:',rmsle(y,y_pred))   
test.columns
train.columns
train.shape
test.shape
a=pd.read_csv('test.csv')
test_pickup_datetime=a['pickup_datetime']
# Instantiate a xgb regressor: xgb

Xgb = XGBRegressor(subsample= 0.1, reg_alpha= 0.08685113737513521, n_estimators= 200, max_depth= 3, learning_rate=0.05, colsample_bytree= 0.7000000000000001, colsample_bynode=0.7000000000000001, colsample_bylevel=0.9000000000000001)



# Fit the regressor to the data

Xgb.fit(X,y)



# Compute and print the coefficients

xgb_features = Xgb.feature_importances_

print(xgb_features)



# Sort feature importances in descending order

indices = np.argsort(xgb_features)[::1]



# Rearrange feature names so they match the sorted feature importances

names = [test.columns[i] for i in indices]



# Creating plot

fig = plt.figure(figsize=(20,10))

plt.title("Feature Importance")



# Add horizontal bars

plt.barh(range(pd.DataFrame(X_train).shape[1]),xgb_features[indices],align = 'center')

plt.yticks(range(pd.DataFrame(X_train).shape[1]), names)

plt.savefig(' xgb1 feature importance')

plt.show()

scores(Xgb)



# Predictions

pred = Xgb.predict(test.values)

pred_results_wrt_date = pd.DataFrame({"pickup_datetime":test_pickup_datetime,"fare_amount" : pred})

pred_results_wrt_date.to_csv("predictions_xgboost.csv",index=False)
pred_results_wrt_date
# Save the model as a pickle in a file 

joblib.dump(Xgb, 'cab_fare_xgboost_model.pkl') 

  

# # Load the model from the file 

# Xgb_from_joblib = joblib.load('cab_fare_xgboost_model.pkl')  
import pandas as pd

test = pd.read_csv("../input/test.csv")

train_cab = pd.read_csv("../input/train_cab.csv")
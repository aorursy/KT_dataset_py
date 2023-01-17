import pandas as pd

import numpy as np

import datetime, warnings, scipy

import matplotlib.pyplot as plt

import math

import category_encoders as ce

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression



warnings.filterwarnings("ignore")
airlines = pd.read_csv('../input/airlines.csv')

airports = pd.read_csv('../input/airports.csv')

flights = pd.read_csv('../input/flights.csv')
# visualize airlines dataset

airlines.head(10)
len(airlines)

print('There are %d airlines in dataset' % len(airlines))
# info airlines dataset

airlines.info()
# describe airlines dataset

airlines.describe()
# visualize airports dataset

airports.head()
len(airports)

print('There are %d airports in dataset' % len(airports))
# info airports dataset

airports.info()
# describe airports dataset

airports.describe()
# visualize flights dataset

flights.head()
flights.tail()
len(flights)

print('There are %d flights in dataset' % len(flights))
# info flights dataset

flights.info()
# describe flights dataset

flights.describe()
# lower case to columns headers



airlines.columns = airlines.columns.str.lower()

airports.columns = airports.columns.str.lower()

flights.columns = flights.columns.str.lower()
flights.isna().sum()
# drop column big amount NA values

flights = flights[flights.columns.difference(['cancellation_reason', 'air_system_delay', 'security_delay',

                                              'airline_delay', 'late_aircraft_delay', 'weather_delay'])]
# drop NA values

flights = flights.dropna()
len(flights)

print('There are %d flights in dataset' % len(flights))
# just keep important columns

# order columns

flights = flights[['month', 'day_of_week', 'airline', 'origin_airport', 'destination_airport',

                   'scheduled_departure', 'departure_delay', 'scheduled_arrival', 'arrival_delay',

                   'scheduled_time', 'elapsed_time', 'distance']]
# new columns time_delay to check on air time delay

flights['time_delay'] = flights.elapsed_time - flights.scheduled_time
# correlation matrix

sns.heatmap(flights.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
# day of week amount of flights

plt.hist(flights['day_of_week'],bins=[1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5])

plt.title("Histogram Days Of Week")

plt.xlim()

plt.show()
# distance distribution

plt.hist(flights['distance'], bins='auto')

plt.title("Histogram distance")

plt.xlim((50, 2000))

plt.show()
# most crowded origin airports ranking

airport_origin_weight = flights.groupby(['origin_airport']).month.count().sort_values(ascending = False)

airport_origin_weight.head(10)
# market shere total flights per airline

airline_weight = flights.groupby(['airline']).month.count().sort_values(ascending = False)/len(flights)

airline_weight
# market share "padding time" flights per airline

df_filter = flights[(flights['departure_delay'] >= 1 ) & (flights['arrival_delay'] <=3 ) 

        & (flights['arrival_delay'] >=-3 ) & (flights['time_delay'] <=-1 )]

airline_weight_filter = df_filter.groupby(['airline']).month.count().sort_values(ascending = False)/len(df_filter)

airline_weight_filter
# rate of padding times per airlines

airline_weight = pd.DataFrame(airline_weight)

airline_weight_filter = pd.DataFrame(airline_weight_filter)

df_padding = pd.merge(airline_weight,airline_weight_filter,on='airline', how='left')

df_padding['rate'] = df_padding.month_y/df_padding.month_x

df_padding.rate.sort_values(ascending = False)
# check unnamed airports

flights.origin_airport.unique()
# drop unnamed airports rows. from 4250000th row airport is unnamed

flights = flights[0:4250000]
# create new dataset flights2. not necessary

flights2 = flights[:]
# hour truncated

flights2['scheduled_departure_hour'] = flights2.scheduled_departure

flights2['scheduled_arrival_hour'] = flights2.scheduled_arrival

flights2['scheduled_departure_hour'] = flights2.scheduled_departure/100

flights2['scheduled_arrival_hour'] = flights2.scheduled_arrival/100

flights2['scheduled_departure_hour'] = np.fix(flights2.scheduled_departure_hour)

flights2['scheduled_arrival_hour'] = np.fix(flights2.scheduled_arrival_hour)
# days_of_week rename values

flights2.day_of_week = flights2.day_of_week.replace([1, 2, 3, 4, 5, 6, 7], ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'])

flights2.day_of_week.unique()
# create dummy variable for days_of_week

dummy = pd.get_dummies(flights2.day_of_week)

flights2 = pd.concat([flights2, dummy], axis=1)

# month rename values

flights2.month = flights2.month.replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

                                      ['january', 'february', 'march', 'april', 'may', 'june', 

                                       'july', 'august', 'september', 'october', 'november', 'december'])

flights2.month.unique()
# create dummy variable for months

dummy = pd.get_dummies(flights2.month)

flights2 = pd.concat([flights2, dummy], axis=1)
# binary encoding to airlines column

flights2['airline2'] = flights2.airline

encoder_airlines = ce.BinaryEncoder(cols=['airline'])

encoder_airlines.fit(flights2)

flights2 = encoder_airlines.transform(flights2)
# binary encoding to origin_airport column

flights2['origin_airport2'] = flights2.origin_airport

encoder_origin_airport = ce.BinaryEncoder(cols=['origin_airport'])

flights2 = encoder_origin_airport.fit_transform(flights2)
# binary encoding to destination_airport column

flights2['destination_airport2'] = flights2.destination_airport

encoder_destination_airport = ce.BinaryEncoder(cols=['destination_airport'])

flights2 = encoder_destination_airport.fit_transform(flights2)
# binary encoding to scheduled_departure_hour column

flights2['scheduled_departure_hour2'] = flights2.scheduled_departure_hour

encoder_scheduled_departure_hour = ce.BinaryEncoder(cols=['scheduled_departure_hour'])

flights2 = encoder_scheduled_departure_hour.fit_transform(flights2)
# binary encoding to scheduled_arrival_hour column

flights2['scheduled_arrival_hour2'] = flights2.scheduled_arrival_hour

encoder_scheduled_arrival_hour = ce.BinaryEncoder(cols=['scheduled_arrival_hour'])

flights2 = encoder_scheduled_arrival_hour.fit_transform(flights2)
flights3 = flights2[flights2.columns.difference(['month', 'day_of_week', 'scheduled_departure', 

                                                'scheduled_arrival', 'elapsed_time', 'time_delay', 'airline2',

                                                'origin_airport2', 'destination_airport2', 'scheduled_departure_hour2',

                                                'scheduled_arrival_hour2', 'departure_delay'])] # departure_delay

# drop arrival_delay outliers

flights3 = flights3[flights3['arrival_delay']<500]
def rmse(y, y_pred):

    return np.sqrt(np.mean(np.square(y - y_pred)))
rmse_baseline = rmse(flights3.arrival_delay,0)

print('The RSME BaseLine is',rmse_baseline)
# standar normalize arrival_delay, distance, schedule_time and departure_delay



std_arrival_delay = flights3.arrival_delay.std()

mean_arrival_delay = flights3.arrival_delay.mean()



flights3.arrival_delay=(flights3.arrival_delay-flights3.arrival_delay.mean())/flights3.arrival_delay.std()

flights3.distance=(flights3.distance-flights3.distance.mean())/flights3.distance.std()

flights3.scheduled_time=(flights3.scheduled_time-flights3.scheduled_time.mean())/flights3.scheduled_time.std()

#flights3.departure_delay=(flights3.departure_delay-flights3.departure_delay.mean())/flights3.departure_delay.std()
# split 30% testing 70% training dataset

train,test=train_test_split(flights3,test_size=0.3,random_state=0)

train_X=train[train.columns.difference(['arrival_delay'])]

train_Y=train['arrival_delay']

test_X=test[test.columns.difference(['arrival_delay'])]

test_Y=test['arrival_delay']
#from sklearn.model_selection import KFold #for K-fold cross validation

#from sklearn.model_selection import cross_val_score #score evaluation

#from sklearn.model_selection import cross_val_predict #prediction

#from sklearn.tree import DecisionTreeRegressor

#from sklearn.linear_model import LinearRegression



kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[] # el % de aciertos en la matriz de confusion

std=[]

classifiers=['DecisionTree', 'LinearRegression']

models=[DecisionTreeRegressor(), LinearRegression()] # Decision Tree Model

for i in models:

    model = i

    cv_result = cross_val_score(model,train_X,train_Y, cv = kfold,scoring = "neg_mean_squared_error")

    cv_result=cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

new_models_dataframe2
final_model=LinearRegression()

final_model.fit(train_X,train_Y) #con la data train_X predice el valor de train_Y

prediction=final_model.predict(test_X)

rsme_denormalize = rmse(prediction*std_arrival_delay+mean_arrival_delay,

                        test_Y*std_arrival_delay+mean_arrival_delay)

print('The RSME of the Linear Regression is',rsme_denormalize)
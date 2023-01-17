import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

import glob

import pprint

import time
print ("Data Sources : ")

print(check_output(["ls", "../input"]).decode("utf8"))



for source in glob.glob("../input/*") :

    print ("\nList the data source from " + source + " : ")

    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint (glob.glob(source + "/*"))



df_train = pd.read_csv("../input/nyc-taxi-trip-duration/train.csv")

df_test = pd.read_csv("../input/nyc-taxi-trip-duration/test.csv")



# Any results you write to the current directory are saved as output.
df_train.info()

df_train.shape



df_test.info()
print ("Is there any null values : ", end = " ")

print( df_train.isnull().values.any()) ## no null values
for i in df_train['vendor_id'].unique():

    print ("Total number of in train " + str(i) + ' : ', end = " ")

    print(df_train[df_train['vendor_id']==i]['id'].count())
print ("Trip distribution with respect to vendor : ")

print( df_train['vendor_id'].value_counts())



## Training data

plt.subplot(2,1,1)

plt.pie(df_train['vendor_id'].value_counts(), 

        labels = df_train['vendor_id'].value_counts(),

       shadow = False,

        startangle = 90,

        explode=(0, 0.05),

       autopct='%1.1f%%')

plt.title("Distribution of trips within vendor for Train Data")

plt.legend(df_train['vendor_id'].unique())

plt.axis('equal')

#plt.tight_layout()



## Testing data

plt.subplot(2,1,2)

plt.pie(df_test['vendor_id'].value_counts(), 

        labels = df_test['vendor_id'].value_counts(),

       shadow = False,

        startangle = 90,

        explode=(0, 0.05),

       autopct='%1.1f%%')

plt.title("Distribution of trips within vendor for Test Data")

plt.legend(df_test['vendor_id'].unique())

plt.axis('equal')

#plt.tight_layout()



plt.show()
import re

from datetime import datetime

def convert_date(date) :

    ## This function convert string to datetime

    match_datetime =  re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', date)

    return datetime.strptime(match_datetime.group(),'%Y-%m-%d %H:%M:%S')



## Apply the function on training set

df_train['pickup_date'] = df_train['pickup_datetime'].apply(lambda x : convert_date(x))

df_train['dropoff_date'] = df_train['dropoff_datetime'].apply(lambda x : convert_date(x))

## Apply function on testing set,

##notice that test dont have dropoff or else this competiton might have been fake

df_test['pickup_date'] = df_test['pickup_datetime'].apply(lambda x : convert_date(x))



## For day

df_train['pickup_day'] = df_train['pickup_date'].apply(lambda x : x.day)

df_test['pickup_day'] = df_test['pickup_date'].apply(lambda x : x.day)



## For month

df_train['pickup_month'] = df_train['pickup_date'].apply(lambda x : x.month)

df_test['pickup_month'] = df_test['pickup_date'].apply(lambda x : x.month)

max(df_train['dropoff_date'] - df_train['pickup_date'])

min(df_train['dropoff_date'] - df_train['pickup_date'])
print ("First trip pickup date is train : " + str(min(df_train['pickup_date'])))

print ("Last trip pickup date in train : " + str(max(df_train['pickup_date'])))

print ("First trip pickup date in test : " + str(min(df_test['pickup_date'])))

print ("Last trip pickup date in test : " + str(max(df_test['pickup_date'])))
## No point to combine Test data

df_p_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')

df_p_2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')

df_p_combined = pd.concat([df_p_1,df_p_2])

df_train_combined = pd.merge(df_train, df_p_combined, on='id', how='left')



## Test data

df_p_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')

df_p_combined = pd.concat([df_p_1,df_p_2])

df_test_combined = pd.merge(df_test, df_p_combined, on='id', how='left')





del df_p_1

del df_p_2

del df_p_combined
df_train_combined.columns

print( "Total null values in train combined set : " + str(df_train_combined.isnull().any().sum()))

print( "Total null values in test combined set : " + str( df_test_combined.isnull().any().sum()))
#nan = df_train_combined[df_train_combined.isnull().any(axis=1)].index

#print("Following index have null values : " + str(nan))



## Lets drop row which has nan

df_train_combined = df_train_combined.dropna()

nan = df_train_combined[df_train_combined.isnull().any(axis=1)].index

print("Following index have null values : " + str(nan))

print( "Total null values in train combined set : " + str(df_train_combined.isnull().any().sum()))

df_train_combined.loc[1133561]
df_train_combined[((df_train_combined['trip_duration'] - df_train_combined['total_travel_time'])/60) > 60]
df_accidents = pd.read_csv('../input/new-york-city-taxi-with-osrm/accidents_2016.csv')

df_accidents.describe()

df_accidents.info()
df_final_train = df_train_combined[['vendor_id','passenger_count','pickup_longitude','pickup_latitude','total_distance','total_travel_time','number_of_steps','store_and_fwd_flag']]
df_final_train = pd.get_dummies(df_final_train)
df_final_train
y = df_train_combined['trip_duration']

X = df_final_train

#X = df_final_train.iloc[:,:]
print(type(y))

print(type(X))
## Use following link for Regression https://medium.com/towards-data-science/simple-and-multiple-linear-regression-in-python-c928425168f9



import statsmodels.api as sm

X = sm.add_constant(X)



model = sm.OLS(y, X).fit()

predictions = model.predict(X)



model.summary()
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X,y)
lm.score(X,y)
import math

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier



import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', None)

df = pd.read_csv('/kaggle/input/los-angeles-metro-bike-share-trip-data/metro-bike-share-trip-data.csv')
df.head()
df.info()
df.isnull().any().any()
df = df.dropna()
sns.distplot(df['Duration'])
def percentiles(column, percentiles):

    for percentile in percentiles:

        print('{0}th percentile = {1}'.format(percentile, np.percentile(column,percentile)))

        

percentiles(df['Duration'],[0,1,5,20,30,40,50,60,70,80,90,100])
df['Duration_log'] = np.log(df['Duration'])

sns.distplot(df['Duration_log'])
print('Trip route categories: ',df['Trip Route Category'].unique())

print('Passholder Types: ',df['Passholder Type'].unique())
sns.catplot(x = 'Passholder Type', y = 'Duration', data = df)
def distance_between_coordinates(lat1, lon1, lat2, lon2):

    earth_radius_km = 6371

    

    lat1 = math.radians(lat1)

    lat2 = math.radians(lat2)

    lon1 = math.radians(lon1)

    lon2 = math.radians(lon2)

    d_lat = lat2 - lat1

    d_lon = lon2 - lon1

    

    a = math.sin(d_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return  earth_radius_km * c
df['Distance'] = df.apply(lambda x: distance_between_coordinates(x['Starting Station Latitude'],x['Starting Station Longitude'],x['Ending Station Latitude'],x['Ending Station Longitude']),axis = 1)
df_copy = df[df.Distance > 0]
df_copy.head()
df_copy.shape
df_copy = pd.concat([df_copy,pd.get_dummies(df_copy['Passholder Type'])],axis = 1)

df_copy.head()


X = df_copy[['Duration_log','Distance']]



def split_predict(X,columns,model=LogisticRegression(),metric = accuracy_score):

    for column in columns:

        y = df_copy[column]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 2000)

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        print('Prediction for column {0}: {1}'.format(column, accuracy_score(y_test,y_pred)))
split_predict(X,['Flex Pass','Monthly Pass','Walk-up'])
#Decision Tree Classifier

split_predict(X,['Flex Pass','Monthly Pass','Walk-up'],DecisionTreeClassifier())
#Extra Trees Classifier

split_predict(X,['Flex Pass','Monthly Pass','Walk-up'],ExtraTreesClassifier())
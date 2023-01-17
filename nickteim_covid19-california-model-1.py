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
import pandas as pd

sample_submission = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")

test = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv")

train = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv")
len(train)

sample_submission.head()
test.head()
train.tail()
#make a heatmap



import folium

from folium import Choropleth, Marker

from folium.plugins import HeatMap, MarkerCluster

m = folium.Map(location=[37, -115], zoom_start=6) 

def embed_map(m, file_name):

    from IPython.display import IFrame

    m.save(file_name)

    return IFrame(file_name, width='100%', height='750px')



#merge test and training data

Full_data = pd.merge(test, train, on=['Lat','Long','Date'])



# Add a heatmap to the base map

HeatMap(data=Full_data[['Lat', 'Long']], radius=11).add_to(m)



# Show the map

embed_map(m, "q_1.html")
#rename therefor the data columns

train.rename(columns={'Province/State':'Province'}, inplace=True)

train.rename(columns={'Country/Region':'Country'}, inplace=True)

train.rename(columns={'ConfirmedCases':'Confirmed'}, inplace=True)
#and we do the same for test set

test.rename(columns={'Province/State':'Province'}, inplace=True)

test.rename(columns={'Country/Region':'Country'}, inplace=True)
from sklearn.preprocessing import LabelEncoder

# creating initial dataframe

bridge_types = ('Lat', 'Date', 'Province', 'Country', 'Long', 'Confirmed',

       'ForecastId', 'Id')

countries = pd.DataFrame(train, columns=['Country'])

# creating instance of labelencoder

labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column

train['Countries'] = labelencoder.fit_transform(train['Country'])



#do the same for test set

test['Countries'] = labelencoder.fit_transform(test['Country'])



#check label encoding 

train['Countries'].head()

train['Date']= pd.to_datetime(train['Date']) 

test['Date']= pd.to_datetime(test['Date']) 
train = train.set_index(['Date'])

test = test.set_index(['Date'])
def create_time_features(df):

    """

    Creates time series features from datetime index

    """

    df['date'] = df.index

    df['hour'] = df['date'].dt.hour

    df['dayofweek'] = df['date'].dt.dayofweek

    df['quarter'] = df['date'].dt.quarter

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['dayofyear'] = df['date'].dt.dayofyear

    df['dayofmonth'] = df['date'].dt.day

    df['weekofyear'] = df['date'].dt.weekofyear

    

    X = df[['hour','dayofweek','quarter','month','year',

           'dayofyear','dayofmonth','weekofyear']]

    return X
create_time_features(train).head()

create_time_features(test).head()
train.head()
train.drop("date", axis=1, inplace=True)

test.drop("date", axis=1, inplace=True)
# train.isnull().sum()
#drop useless columns for train and test set

train.drop(['Country'], axis=1, inplace=True)

train.drop(['Province'], axis=1, inplace=True)
test.drop(['Country'], axis=1, inplace=True)

test.drop(['Province'], axis=1, inplace=True)
from sklearn.tree import DecisionTreeRegressor  

regressor = DecisionTreeRegressor(random_state = 0) 
# import xgboost as xgb

# from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error



# reg= xgb.XGBRegressor(n_estimators=1000)
train.head()
# features that will be used in the model

x = train[['Lat', 'Long','Countries','dayofweek','month','dayofyear','weekofyear']]

y1 = train[['Confirmed']]

y2 = train[['Fatalities']]

x_test = test[['Lat', 'Long','Countries','dayofweek','month','dayofyear','weekofyear']]
x.head()
#use model on data 

regressor.fit(x,y1)

predict_1 = regressor.predict(x_test)

predict_1 = pd.DataFrame(predict_1)

predict_1.columns = ["Confirmed_predict"]
predict_1.head()
#use model on data 

regressor.fit(x,y2)

predict_2 = regressor.predict(x_test)

predict_2 = pd.DataFrame(predict_2)

predict_2.columns = ["Death_prediction"]

predict_2.head()
# plot = plot_importance(regressor, height=0.9, max_num_features=20)
Samle_submission = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv")

Samle_submission.columns

submission = Samle_submission[["ForecastId"]]
Final_submission = pd.concat([predict_1,predict_2,submission],axis=1)

Final_submission.head()
Final_submission.columns = ['ConfirmedCases', 'Fatalities', 'ForecastId']

Final_submission = Final_submission[['ForecastId','ConfirmedCases', 'Fatalities']]



Final_submission["ConfirmedCases"] = Final_submission["ConfirmedCases"].astype(int)

Final_submission["Fatalities"] = Final_submission["Fatalities"].astype(int)
Final_submission.head()
Final_submission.to_csv("submission.csv",index=False)

print('Model ready for submission!')
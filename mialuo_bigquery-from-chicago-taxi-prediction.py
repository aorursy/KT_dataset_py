from google.cloud import bigquery

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model,metrics,tree

from sklearn.ensemble import RandomForestRegressor



client = bigquery.Client()
chicago_data_ref = client.dataset('chicago_taxi_trips',project='bigquery-public-data')

chicago_datasets = client.get_dataset(chicago_data_ref)

[i.table_id for i in client.list_tables(chicago_data_ref)]
chicago_table = client.get_table(chicago_datasets.table('taxi_trips'))

chicago_table.schema
client.query("""SELECT count(*)

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE trip_miles>0""").to_dataframe()
n = 100000

query_string = """SELECT *

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE trip_miles>0

ORDER BY RAND()

LIMIT ?"""



job_config = bigquery.QueryJobConfig(

    query_parameters=[

        bigquery.ScalarQueryParameter(None, "INT64", n)

    ]

)

result  = client.query(query_string,job_config).to_dataframe()
result.describe()
total = (result.isnull().sum()/n).sort_values(ascending = False)

total
columns_keep = [i for i in result.columns if i not in ["dropoff_census_tract","pickup_census_tract","company","tolls"]]

result = result[columns_keep]

for i in ["dropoff_community_area","dropoff_longitude","dropoff_location","dropoff_latitude",\

            "pickup_community_area","pickup_latitude","pickup_longitude","pickup_location"]:

    result = result[result[i].notnull()]

len(result)
for i in ["trip_seconds","trip_miles","fare"]:

    q1 = result[i].quantile(0.25)

    q3 = result[i].quantile(0.75)

    result = result[result[i] < q3 + 5*(q3-q1)]
result.describe()
fig, axs = plt.subplots(1,2)

fig.set_size_inches(10, 4)

axs[0].set_title('density plot of tripmiles')

sns.distplot(result['trip_miles'],ax = axs[0])

axs[1].set_title('density plot of fare')

sns.distplot(result['fare'],ax = axs[1])
result.plot.scatter(x = 'trip_miles',y='fare')
result['high_slope']= result['fare']/result['trip_miles'] > 10

groups = result.groupby('high_slope')

sns.pairplot(x_vars = ['trip_miles'],y_vars=['trip_seconds'],data=result,hue = 'high_slope',height=6)
sns.boxplot([i.hour for i in result['trip_start_timestamp']],result['trip_miles']/result['trip_seconds'])

plt.ylim(0,0.02)
### split the data into training data and testing data, and standardize columns

x = result[['trip_miles','trip_seconds']].copy()

x["interaction"] = result['trip_miles']*result["trip_seconds"]

y = pd.DataFrame(result['fare']).copy()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1212121)



scx = StandardScaler()

scy = StandardScaler()

x_train = scx.fit_transform(x_train)

x_test = scx.transform(x_test)



y_train = scy.fit_transform(y_train)

y_test = scy.transform(y_test)





### Build linear regression model

m_linear = linear_model.LinearRegression()

m_linear.fit(x_train,y_train)

### R^2

linear_train_r2 = m_linear.score(x_train,y_train)

linear_test_r2 = m_linear.score(x_test,y_test)



### MSE

linear_train_error = metrics.mean_squared_error(m_linear.predict(x_train),y_train)

linear_test_error = metrics.mean_squared_error(m_linear.predict(x_test),y_test)



linear_metrics =  pd.DataFrame([[linear_train_r2,linear_test_r2,linear_train_error,linear_test_error]])





linear_metrics.columns = ["trainingR2","testingR2","traingMSE","testingMSE"]

linear_metrics
m_linear.coef_

#m_linear.intercept_
### we only know 'trip_start_timestamp','pickup_community_area','dropoff_community_area','pickup_latitude', 'pickup_longitude', 

### 'dropoff_latitude','dropoff_longitude' before trip starts



### a heuristic estimator for trip_miles will be the Euclidean distance between the pickup_location and dropoff_location



### Let's fit a very simple regression using the rough estimation of trip_miles

x = pd.DataFrame()

x['distance'] = pow(pow((result["pickup_latitude"] - result["dropoff_latitude"]),2) + pow((result["pickup_longitude"] - result["dropoff_longitude"]),2),.5)

y = pd.DataFrame(result['fare']).copy()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1212121)



scx = StandardScaler()

scy = StandardScaler()

x_train = scx.fit_transform(x_train)

x_test = scx.transform(x_test)



y_train = scy.fit_transform(y_train)

y_test = scy.transform(y_test)





### Build linear regression model

m_linear_1 = linear_model.LinearRegression()

m_linear_1.fit(x_train,y_train)

### R^2

linear_train_r2 = m_linear_1.score(x_train,y_train)

linear_test_r2 = m_linear_1.score(x_test,y_test)



### MSE

linear_train_error = metrics.mean_squared_error(m_linear_1.predict(x_train),y_train)

linear_test_error = metrics.mean_squared_error(m_linear_1.predict(x_test),y_test)



linear_metrics =  pd.DataFrame([[linear_train_r2,linear_test_r2,linear_train_error,linear_test_error]])





linear_metrics.columns = ["trainingR2","testingR2","traingMSE","testingMSE"]

linear_metrics
### we will need the starting time of the trip as categorical variable

x = pd.DataFrame()

x['distance'] = pow(pow((result["pickup_latitude"] - result["dropoff_latitude"]),2) + pow((result["pickup_longitude"] - result["dropoff_longitude"]),2),.5)

x['trip_start_hour'] = [i.hour for i in result['trip_start_timestamp']]

x['trip_start_hour'] = x['trip_start_hour'].astype('category')



### one-hot encoding

x = pd.get_dummies(x)



### interactions between continuous variable and categorical variable

for i in range(24):

    x[('trip_start_hour_'+str(i)+'dist')] =  x['trip_start_hour_'+str(i)] * x['distance']



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1212121)

scx = StandardScaler()

scy = StandardScaler()

x_train = scx.fit_transform(x_train)

x_test = scx.transform(x_test)



y_train = scy.fit_transform(y_train)

y_test = scy.transform(y_test)
%%time

### Elastic-Net regression since we have a lot of features

m_elas = linear_model.ElasticNetCV(cv = 10,l1_ratio = [.1, .5, .7, .9, .95, .99])### 10fold cross-validation is used for choosing tunning parameter

m_elas.fit(x_train,y_train.ravel())



elas_train_r2 = m_elas.score(x_train,y_train)

elas_test_r2 = m_elas.score(x_test,y_test)

elas_train_error = metrics.mean_squared_error(m_elas.predict(x_train),y_train)

elas_test_error = metrics.mean_squared_error(m_elas.predict(x_test),y_test)



elas_metrics = pd.DataFrame([[elas_train_r2,elas_test_r2,elas_train_error,elas_test_error]])



elas_metrics.columns = ["trainingR2","testingR2","traingMSE","testingMSE"]

elas_metrics
#### The features selected by Elastic-Net

elas_features = x.columns[m_elas.coef_!=0]

elas_features
### all avaliable informations

x = result[['pickup_community_area', 'dropoff_community_area','pickup_latitude', 'pickup_longitude',\

            'dropoff_latitude', 'dropoff_longitude']].copy()

x['trip_start_hour'] = [i.hour for i in result['trip_start_timestamp']]

x['trip_start_hour'] = x['trip_start_hour'].astype('category')

x['pickup_community_area'] = x['pickup_community_area'].astype('category')

x['dropoff_community_area'] = x['dropoff_community_area'].astype('category')



y = pd.DataFrame(result['fare']).copy()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1212121)

scx = StandardScaler()

scy = StandardScaler()

x_train = scx.fit_transform(x_train)

x_test = scx.transform(x_test)



y_train = scy.fit_transform(y_train)

y_test = scy.transform(y_test)





### CART

m_tree = tree.DecisionTreeRegressor()

m_tree = m_tree.fit(x_train,y_train)



tree_train_error = metrics.mean_squared_error(m_tree.predict(x_train),y_train)

tree_test_error = metrics.mean_squared_error(m_tree.predict(x_test),y_test)



tree_metrics = pd.DataFrame([[tree_train_error,tree_test_error]])



tree_metrics.columns = ["traingMSE","testingMSE"]

tree_metrics
%%time

m_rf = RandomForestRegressor()

m_rf = m_rf.fit(x_train,y_train.ravel())



rf_train_error = metrics.mean_squared_error(m_rf.predict(x_train),y_train)

rf_test_error = metrics.mean_squared_error(m_rf.predict(x_test),y_test)



rf_metrics = pd.DataFrame([[rf_train_error,rf_test_error]])



rf_metrics.columns = ["traingMSE","testingMSE"]

rf_metrics
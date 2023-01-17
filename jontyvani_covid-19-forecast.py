##################################################################

# LIBRARIES

##################################################################



import warnings

from sklearn.cluster import KMeans

import csv

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn_pandas import DataFrameMapper

from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation

import matplotlib.dates as dates

from cycler import cycler

from scipy.ndimage.filters import gaussian_filter1d

import fbprophet

import plotly.express as px

from io import StringIO

import requests

from datetime import timedelta



pd.set_option("max_rows", 1000)

pd.set_option("max_columns", 30)

warnings.filterwarnings("ignore")



##################################################################

# VARIABLES

##################################################################



calculate_clusters = True

calculate_cluster_forecast = True

plot_clusters = True

plot_cluster_forecast_individual = True

plot_cluster_forecast = True



train_history_days = 16

forecast_future_days = 60

confirmed_cases_min = 0



##################################################################

# READ DATA SOURCES

##################################################################



headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}

url="https://gitlab.com/Shankspanks/covid-19-forecast/-/raw/master/train.csv"

s=requests.get(url, headers= headers).text

train = pd.read_csv(StringIO(s), sep=",", index_col=['Id'])



url="https://gitlab.com/Shankspanks/covid-19-forecast/-/raw/master/country_metrics.csv"

s=requests.get(url, headers= headers).text

metrics = pd.read_csv(StringIO(s), sep=",", index_col=['GeoID'])



url="https://gitlab.com/Shankspanks/covid-19-forecast/-/raw/master/test.csv"

s=requests.get(url, headers= headers).text

test = pd.read_csv(StringIO(s), sep=",")



url="https://gitlab.com/Shankspanks/covid-19-forecast/-/raw/master/submission.csv"

s=requests.get(url, headers= headers).text

submission = pd.read_csv(StringIO(s), sep=",")



country_lookups = metrics[['Geo']]

covidclusters = metrics[['Movement', 'Density', 'BCG']]  # Subsetting the data



# clean

train["Province_State"].fillna("", inplace=True)

test["Province_State"].fillna("", inplace=True)

train["Geo"] = train['Country_Region'] + '_' + train['Province_State']

test["Geo"] = test['Country_Region'] + '_' + test['Province_State']

train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])

test['ForecastId_'] = test.index

##################################################################

# CLUSTER COUNTRIES

##################################################################



submission_date_min = test['Date'].min()

train_date_max = train['Date'].max()

future_date = train_date_max + timedelta(days=forecast_future_days)

history_date = train_date_max - timedelta(days=train_history_days)

train = train[

    train["Date"] >= history_date].reset_index()



#scale the data

mapper = DataFrameMapper([(covidclusters.columns, StandardScaler())])

scaled_features = mapper.fit_transform(covidclusters.copy(), 4)

country_clusters_out = pd.DataFrame(

    scaled_features, index=covidclusters.index, columns=covidclusters.columns)



def doAffinity(X):

    model = AffinityPropagation(

        damping=0.5, max_iter=250, affinity='euclidean',verbose=True)

    model.fit(X)

    clust_labels2 = model.predict(X)

    cent2 = model.cluster_centers_

    return (clust_labels2, cent2)



if calculate_clusters == True:

    clust_labels2, cent2 = doAffinity(country_clusters_out)

    affinity = pd.DataFrame(clust_labels2,columns={'cluster'}).reset_index()

    affinity['GeoID'] = affinity['index']+1

    country_clusters_out = pd.merge(country_clusters_out, affinity, how='left', left_on=[

                       'GeoID'], right_on=['GeoID'], left_index=True)

    country_clusters_out = pd.merge(country_clusters_out, country_lookups, how='left', left_on=[

                       'GeoID'], right_on=['GeoID'], left_index=True)

    country_clusters_out.to_csv('country_clusters_out.csv')

else:

    url="https://gitlab.com/Shankspanks/covid-19-forecast/-/raw/master/country_clusters_out.csv"

    s=requests.get(url, headers= headers).text

    country_clusters_out = pd.read_csv(StringIO(s), sep=",")

##################################################################

# PLOT COUNTRY CLUSTERS IN 3D

##################################################################



if plot_clusters == True:

    fig = px.scatter_3d(country_clusters_out, x='Movement', y='Density', z='BCG',

                  color='cluster', hover_data =['Geo'])

    fig.show()

    



display(country_clusters_out)
##################################################################

# FORECAST EACH CLUSTER

##################################################################



df_train = pd.merge(country_clusters_out, train, how='left', left_on=[

                    'Geo'], right_on=['Geo'], left_index=True)

df_forecast_confirmed = df_train[[

    'Geo', 'Date', 'cluster', 'ConfirmedCases', 'Fatalities']].reset_index()

df_forecast_confirmed = df_forecast_confirmed[df_forecast_confirmed['ConfirmedCases'] >= confirmed_cases_min]



cluster_curves = df_forecast_confirmed.groupby(

    ['Date', 'cluster']).sum().reset_index()

cluster_curves = cluster_curves.sort_values(

    by=['cluster', 'Date'], ascending=True).reset_index()

cluster_curves = cluster_curves[

    ['Date', 'cluster', 'ConfirmedCases', 'Fatalities']].reset_index()



#create new columns called ConfirmedCases_New, Fatalities_New

cluster_curves['ConfirmedCases_Lag'] = cluster_curves['ConfirmedCases'].shift(periods=1, fill_value=0)

cluster_curves['Fatalities_Lag'] = cluster_curves['Fatalities'].shift(periods=1)

cluster_curves['cluster_Lag'] = cluster_curves['cluster'].shift(periods=1)

cluster_curves['ConfirmedCases_New'] = cluster_curves['ConfirmedCases']-cluster_curves['ConfirmedCases_Lag']

cluster_curves['Fatalities_New'] = cluster_curves['Fatalities']-cluster_curves['Fatalities_Lag']

cluster_curves.loc[cluster_curves['cluster'] != cluster_curves['cluster_Lag'], 'ConfirmedCases_New'] = 0

cluster_curves.loc[cluster_curves['cluster'] != cluster_curves['cluster_Lag'], 'Fatalities_New'] = 0



cluster_curves = cluster_curves.rename(columns={'Date': 'ds'})

cluster_curves['ds'] = pd.to_datetime(cluster_curves['ds'])



clusterMax = cluster_curves.cluster.max()

count = 0

cluster_forecast_out = cluster_curves[['ds','cluster','ConfirmedCases','Fatalities','ConfirmedCases_New','Fatalities_New']] 

current_date = cluster_forecast_out['ds'].max()



#trim one zero period

cluster_curves = cluster_curves[

    cluster_curves["ds"] > history_date].reset_index()



if calculate_cluster_forecast == True:

    while count <= clusterMax:

        # confirmed

        CCModel = fbprophet.Prophet(changepoint_prior_scale=0.05)

        cluster_curves['y'] = cluster_curves['ConfirmedCases_New']

        temp1 = cluster_curves.loc[(cluster_curves.cluster == count)]

        CCModel.fit(temp1)

        CCForecast = CCModel.make_future_dataframe(periods=forecast_future_days, freq='D')

        CCForecast = CCModel.predict(CCForecast)

        merge1 = CCForecast.copy()

        merge1 = merge1.loc[(merge1.ds > current_date)]

        # new cases can't go below zero

        merge1.loc[merge1['yhat'] < 0, ['yhat']] = 0

        #plot

        if plot_cluster_forecast_individual == True:

            CCModel.plot(CCForecast, xlabel = 'Date', ylabel = 'new confirmed cases')

            plt.title('cluster: '+ str(count))

            plt.show()



        CCModel2 = fbprophet.Prophet(changepoint_prior_scale=0.05)

        cluster_curves['y'] = cluster_curves['Fatalities_New']

        CCModel2.fit(cluster_curves.loc[(cluster_curves.cluster == count)] )

        CCForecast2 = CCModel2.make_future_dataframe(periods=forecast_future_days, freq='D')

        CCForecast2 = CCModel2.predict(CCForecast2)

        merge2 = CCForecast2.copy()    

        merge2 = merge2.loc[(merge2.ds > current_date)]

        # new fatalities can't go below zero

        merge2.loc[merge2['yhat'] < 0, ['yhat']] = 0

        #plot

        if plot_cluster_forecast_individual == True:

            CCModel2.plot(CCForecast2, xlabel = 'Date', ylabel = 'new fatalites')

            plt.title('cluster: '+ str(count) )

            plt.show()



        #print(CCForecast[['ds','yhat']])

        merge1['cluster'] = count

        merge1['ConfirmedCases_New'] = merge1['yhat']

        merge2['Fatalities_New'] = merge2['yhat']



        forecast = pd.merge(merge1, merge2, how = 'inner', left_on = 'ds', right_on = 'ds')



        forecast = forecast[['ds','cluster','ConfirmedCases_New','Fatalities_New']] 

        forecast['ConfirmedCases'] = -1

        forecast['Fatalities'] = -1

        cluster_forecast_out = cluster_forecast_out.append(forecast)

        #print(cluster_forecast_out)



        count += 1  # This is the same as count = count + 1



    cluster_forecast_out = cluster_forecast_out.rename(columns={'ds': 'Date'})

    cluster_forecast_out = cluster_forecast_out.sort_values(['cluster', 'Date'], ascending=[True, True]).reset_index()

    confirmed = -1

    fatality = -1



    for index, row in cluster_forecast_out.iterrows():

        if row['ConfirmedCases'] == -1:

            row['ConfirmedCases'] = confirmed + row['ConfirmedCases_New']

            row['Fatalities'] = fatality + row['Fatalities_New']

            cluster_forecast_out.at[index,'ConfirmedCases']=row['ConfirmedCases']

            cluster_forecast_out.at[index,'Fatalities']=row['Fatalities']

            #print(row)

        #else:

            #print("skip")

        confirmed = row['ConfirmedCases']

        fatality = row['Fatalities']

    

    cluster_forecast_out = cluster_forecast_out[

    cluster_forecast_out["Date"] > history_date].reset_index()

    cluster_forecast_out.to_csv('cluster_forecast_out.csv')

else:

    url="https://gitlab.com/Shankspanks/covid-19-forecast/-/raw/master/cluster_forecast_out.csv"

    s=requests.get(url, headers= headers).text

    cluster_forecast_out = pd.read_csv(StringIO(s), sep=",")



cluster_curves = cluster_curves.rename(columns={'ds': 'Date'})

cluster_curves['Date'] = pd.to_datetime(cluster_curves['Date'])
##################################################################

# PLOT CLUSTER FORECASTS

##################################################################



charr = cluster_forecast_out

charr.Date = pd.to_datetime(charr.Date)

charr = charr[['ConfirmedCases',	'Fatalities', 'cluster','Date']]

charr.set_index('Date')

# confirmed cases chart

count = 0

fig = plt.figure()

fig.suptitle('Confirmed Cases', fontsize=10)

while count <= 11:

    temp1 = charr.loc[(charr.cluster == count)][['Date','ConfirmedCases',	'Fatalities']]

    temp1.set_index('Date')

    ax = fig.add_subplot(111)

    ax.tick_params(axis='both', which='major', labelsize=5) 

    ax.tick_params(axis='both', which='minor', labelsize=5)

    ax.plot(temp1['Date'],temp1['ConfirmedCases'],

        label='cluster: '+str(count) , fillstyle='none')

    plt.legend(loc=2, prop={'size': 5})

    #ax.set_yscale('log')

    plt.xticks(rotation=90)

    count += 1  

if plot_cluster_forecast == True:    

    plt.show()

# Fatalities

count = 0

fig = plt.figure()

fig.suptitle('Fatalities', fontsize=10)

while count <= 11:

    temp1 = charr.loc[(charr.cluster == count)][['Date','ConfirmedCases',	'Fatalities']]

    temp1.set_index('Date')

    ax = fig.add_subplot(111)

    ax.tick_params(axis='both', which='major', labelsize=5) 

    ax.tick_params(axis='both', which='minor', labelsize=5)

    ax.plot(temp1['Date'],temp1['Fatalities'],

        label='cluster: '+str(count) , fillstyle='none')

    plt.legend(loc=2, prop={'size': 5})

    #ax.set_yscale('log')

    plt.xticks(rotation=90)

    count += 1  

if plot_cluster_forecast == True: 

    plt.show()
##################################################################

# GET COUNTRY TO CLUSTER RATIOS

##################################################################



# CREATE SITE CLUSTER CONFIRMED RATIO AND FATALITY RATIO INTERFACE

# MOST RECENT X DAYS OF CLUSTER / X DAYS OF COUNTRY

# SAVE AS COUNTRY - CLUSTER - RATIO_CONFIRMED - RATIO_FATALITY



# countries interface

countries_interface = pd.DataFrame(

    train.Geo.unique(), columns=['Geo']).reset_index()



countries_interface = pd.merge(countries_interface, country_clusters_out, how='left', left_on=[

    'Geo'], right_on=['Geo'], left_index=True)



cluster_summary = cluster_curves[(

    cluster_curves["Date"] > history_date) & (cluster_curves["Date"] <= train_date_max)]



cluster_summary_agg = cluster_summary.groupby(

    ['cluster']).sum().reset_index()



train_summary = train[

    train["Date"] > history_date].reset_index()



train_summary_agg = train_summary.groupby(

    ['Geo']).sum().reset_index()



countries_interface = pd.merge(countries_interface, cluster_summary_agg[['cluster', 'ConfirmedCases', 'Fatalities']], how='left', left_on=[

    'cluster'], right_on=['cluster'], left_index=True)



countries_interface = pd.merge(countries_interface, train_summary_agg[['Geo', 'ConfirmedCases', 'Fatalities']], how='left', left_on=[

    'Geo'], right_on=['Geo'], left_index=True)



display(countries_interface)
##################################################################

# SPLIT CLUSTER FORECASTS TO COUNTRIES

##################################################################



# stick infintismal for zero

countries_interface[countries_interface["ConfirmedCases_y"] == 0].ConfirmedCases_y = 0.000000001

countries_interface[countries_interface["Fatalities_y"] == 0].Fatalities_y = 0.000000001



countries_interface['RATIO_CONFIRMED'] = countries_interface.ConfirmedCases_y / countries_interface.ConfirmedCases_x

countries_interface['RATIO_FATALITY'] = countries_interface.Fatalities_y / countries_interface.Fatalities_x

countries_interface = countries_interface[[

    'Geo', 'cluster', 'RATIO_CONFIRMED', 'RATIO_FATALITY']]



#countries_interface.to_csv('countries_interface.csv')



country_forecast_out = pd.merge(cluster_forecast_out, countries_interface, on=['cluster'])



country_forecast_out = country_forecast_out[[

    'Geo','Date','cluster','ConfirmedCases','Fatalities','RATIO_CONFIRMED','RATIO_FATALITY','ConfirmedCases_New', 'Fatalities_New']]

country_forecast_out['ConfirmedCases'] = country_forecast_out['ConfirmedCases'] * country_forecast_out['RATIO_CONFIRMED']

country_forecast_out['Fatalities'] = country_forecast_out['Fatalities'] * country_forecast_out['RATIO_FATALITY']

country_forecast_out['ConfirmedCases_New'] = country_forecast_out['ConfirmedCases_New'] * country_forecast_out['RATIO_CONFIRMED']

country_forecast_out['Fatalities_New'] = country_forecast_out['Fatalities_New'] * country_forecast_out['RATIO_FATALITY']

country_forecast_out.to_csv('country_forecast_out.csv')



display(country_forecast_out)
##################################################################

# PREPARE A SUBMISSION FOR COMPETITION

##################################################################



subsy = pd.merge(test, country_forecast_out, how = 'left', left_on = ['Date','Geo'], right_on = ['Date','Geo'])

subsy = subsy[['ForecastId_','ConfirmedCases','Fatalities']]

subsy = subsy.rename(columns={'ForecastId_': 'ForecastId'})

new_row = {'ForecastId':13459, 'ConfirmedCases':100, 'Fatalities':12}

subsy = subsy.append(new_row, ignore_index=True)

subsy2 = pd.merge(submission, subsy, how = 'left', left_on = ['ForecastId'], right_on = ['ForecastId'])

subsy2 = subsy2[['ForecastId','ConfirmedCases_y','Fatalities_y']]

subsy2 = subsy2.rename(columns={'ConfirmedCases_y': 'ConfirmedCases'})

subsy2 = subsy2.rename(columns={'Fatalities_y': 'Fatalities'})



header = ['ForecastId','ConfirmedCases','Fatalities']

subsy2.to_csv('submission.csv', columns = header, index=False)



display(subsy2)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from  bq_helper import BigQueryHelper as bqh

bqa = bqh('bigquery-public-data','google_analytics_sample')
QUERY = """

    SELECT

        date as Date,

        SUM(totals.transactionRevenue)/1e6 as Revenue

    FROM 

      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga

    WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'

    GROUP BY Date

    ORDER BY Date ASC

"""



data = bqa.query_to_pandas(QUERY)

data.Revenue=data.Revenue.fillna(0)

data.Date = data.Date.apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))



print('Days of data: ',len(data))

print('Total Revenue: $',end='')

print("{:,}".format(data.Revenue.sum()))

data.head(14)
# Holt Forecasting model using the statsmodels package.



from statsmodels.tsa.holtwinters import Holt

import matplotlib.ticker as mtick

plt.rcParams['axes.facecolor'] = 'whitesmoke'



for t in [300,200,60]:

    # Model creation and prediction

    y = data.Revenue[:-t].values

    model = Holt(y).fit(smoothing_level = 0.3,smoothing_slope = 0.1)

    y_pred = model.forecast(t)



    # Setting the data columns

    data['Forecast'] = np.append(np.full(len(y), np.nan),y_pred)

    data.Forecast = data.Forecast.values.clip(min=0)

    data['Previous Revenue'] = np.append(y,np.full(len(y_pred), np.nan))

    data['Previous Revenue'] = data['Previous Revenue'].values.clip(min=0)

    data['Actual Revenue'] = np.append(np.full(len(y), np.nan),data.Revenue[-t:].values)



    # Plotting the results

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    data.plot(x='Date',y=['Previous Revenue','Actual Revenue','Forecast'],style=['-','c-','k--'],ax=ax,xlim=([data.Date.iloc[0],data.Date.iloc[-1]]))

    plt.title('Total Revenue forecast of the last '+str(t)+' days of data')

    fmt = '${x:,.0f}'

    tick = mtick.StrMethodFormatter(fmt)

    ax.yaxis.set_major_formatter(tick)

    plt.ylabel('Daily Revenue')

    plt.show()

QUERY = """

    SELECT

        trafficSource.source as Source,

        SUM(totals.transactionRevenue)/1e6 as Revenue,

        AVG(totals.timeOnSite) as Time,

        AVG(totals.transactionRevenue)/1e6 as AverageRevenue

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE

        _TABLE_SUFFIX BETWEEN '20160801'AND '20170801' AND totals.transactions > 0

    GROUP BY Source

    ORDER BY Revenue DESC

"""



datarevenue = bqa.query_to_pandas(QUERY)



QUERY = """

    SELECT

        trafficSource.source as Source,

        AVG(totals.timeOnSite) as Time

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE

        _TABLE_SUFFIX BETWEEN '20160801'AND '20170801' AND totals.transactions IS NULL

    GROUP BY Source

    ORDER BY TIME DESC

"""



datanorevenue = bqa.query_to_pandas(QUERY)



datarevenue.drop(columns=['Time']).head(10)
datarevenue = datarevenue.sort_values(by='Time')

plt.figure(figsize=(6,10))

plt.barh(y = datarevenue.Source.values, width = datarevenue.Time.values)

plt.ylabel('Traffic Source')

plt.xlabel('Average time on site (Seconds)')

plt.title('Time on site for sessions with a transaction')

plt.show()
QUERY = """  

  SELECT 

        fullVisitorId,

        visitId,

        trafficSource.source as TrafficSource,

        device.browser as Browser,

        device.deviceCategory as Device,

        geoNetwork.subContinent as Location,

        hits.page.pagePath,

        totals.timeOnSite as y

  FROM 

      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,

      unnest(hits) as hits

  WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20161001' and hits.time = 0

  ORDER BY fullVisitorId,visitStartTime

  """



entry = bqa.query_to_pandas(QUERY)

entry.y = entry.y.fillna(0)



X=entry.drop(columns=['fullVisitorId','visitId','y']).values

y = entry.y.values.reshape(-1,1) > 0



from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(X)

X = enc.transform(X).toarray()



from sklearn.model_selection import train_test_split

idx,idy,X_train,X_test,y_train,y_test = train_test_split(list(range(len(entry))),X,y,test_size=0.33)



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train.ravel())

y_pred = model.predict(X_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)



classes=['Does not Bounce','Bounce']

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()

im = ax.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)

ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),xticklabels=classes, yticklabels=classes,title='Bounce prediction results',ylabel='True label', xlabel='Predicted label')



fmt = '.2f'

thresh = cm.max() / 2.

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center",color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()



entry['TestSet'] = [1 if i in idy else 0 for i in range(0,len(entry))]

entry['rfc'] = model.predict(X)*1
QUERY = """  

  SELECT 

        fullVisitorId,

        visitId,

        totals.timeOnSite,

        hits.page.pagePath,

        visitStartTime as vst

  FROM 

      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,

      unnest(hits) as hits

  WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20161001' AND isExit IS TRUE

  ORDER BY fullVisitorID, visitStartTime

  """



leave = bqa.query_to_pandas(QUERY)



data = pd.merge(entry, leave,  how='left', left_on=['fullVisitorId','visitId'], right_on = ['fullVisitorId','visitId']).dropna(subset = ['pagePath_x','pagePath_y'])

data.rename(columns={'pagePath_x':'Entry Path','pagePath_y':'Exit Path'}, inplace=True)

data[['visitId','TrafficSource','Browser','Device','Location','Entry Path','Exit Path']].head(20)
data = data.groupby('Entry Path')['Exit Path'].apply(lambda x: x.value_counts(normalize=True).head(1)*100)

data = data.reset_index().rename(index=str, columns={"level_1": "Most likely exit page","Exit Path": "Proportion (%)"})

data.head(20)
QUERY =  """

    SELECT

        fullVisitorId,

        trafficSource.source as TrafficSource,

        device.browser as Browser,

        device.deviceCategory as Device,

        geoNetwork.country as Country,

        SUM(totals.transactionRevenue)/1e6 as Revenue

    FROM 

      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga

    WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'

    GROUP BY fullVisitorId,TrafficSource,Browser,Device,Country

    ORDER BY Revenue DESC

    LIMIT 10

"""



data = bqa.query_to_pandas(QUERY)

data
QUERY =  """

    SELECT

        trafficSource.source as TrafficSource,

        device.browser as Browser,

        device.deviceCategory as Device,

        geoNetwork.country as Country,

        SUM(totals.transactionRevenue)/1e6 as Revenue

    FROM 

      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga

    WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'

    GROUP BY TrafficSource,Browser,Device,Country

    ORDER BY Revenue DESC

    LIMIT 10

"""



data = bqa.query_to_pandas(QUERY)

data
QUERY = """

    SELECT

        fullVisitorId,

        MAX(totals.transactionRevenue)/1e6 as Revenue

    FROM 

        `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga

        WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20170201' AND totals.transactionRevenue > 0

    GROUP BY fullVisitorId,visitId

"""



data = bqa.query_to_pandas(QUERY)

data[data.duplicated(subset=['fullVisitorId'])].groupby('fullVisitorId').sum().reset_index()



QUERY = """

    SELECT DISTINCT

        fullVisitorId

    FROM 

        `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga

        WHERE _TABLE_SUFFIX BETWEEN '20170202' AND '20170801' AND totals.transactionRevenue > 0

    GROUP BY fullVisitorId,visitId

"""



churn = bqa.query_to_pandas(QUERY)

data['Churn']=[0 if x in churn.fullVisitorId.values else 1 for x in data.fullVisitorId.values]



QUERY = """

    SELECT DISTINCT

        fullVisitorId

    FROM 

        `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga

        WHERE _TABLE_SUFFIX BETWEEN '20170202' AND '20170801'

"""



siteview = bqa.query_to_pandas(QUERY)

data['Return']=[1 if x in siteview.fullVisitorId.values else 0 for x in data.fullVisitorId.values]
data.groupby('Churn').Revenue.mean()
print('Proportion who visited the site in the last 6 months of data: '+str(round(100*data.Return.mean(),1))+'%')

print('Proportion who purchased in the last 6 months of data: '+str(round(100*(1-data.Churn.mean()),1))+'%')
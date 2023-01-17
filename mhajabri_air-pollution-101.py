# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import pandas as pd
import numpy as np
from google.cloud import bigquery
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pollutants = ['o3','co','no2','so2','pm25_frm']
QUERY2016 = """
    SELECT
        pollutant.county_name AS County, AVG(pollutant.aqi) AS AvgAQI_pollutant
    FROM
      `bigquery-public-data.epa_historical_air_quality.pollutant_daily_summary` as pollutant
    WHERE
      pollutant.poc = 1
      AND EXTRACT(YEAR FROM pollutant.date_local) = 2016
    GROUP BY 
      pollutant.county_name
"""

df_2016 = None
for elem_g in pollutants : 
    query = QUERY2016.replace("pollutant", elem_g)
    temp = bq_assistant.query_to_pandas(query).set_index('County')
    df_2016 = pd.concat([df_2016, temp], axis=1, join='outer')
df_2016=df_2016.apply(lambda x: x.fillna(x.mean()),axis=0)

df_2016.sample(10,random_state=42)
from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2,n_iter=2000,perplexity=35,random_state=5).fit_transform(df_2016)
df_tsne = pd.DataFrame(X_tsne)
df_tsne['County'] = list(df_2016.index)
df_tsne = df_tsne.set_index('County')
df_tsne.columns = ['ax1', 'ax2']

df_tsne.plot(kind='scatter', x='ax1', y='ax2',figsize=(10,8));

#c1 : ax1=[10 , 35]  ax2=[-20 , 10] 
#c2 : ax1=[-30 , -5]  ax2= [-10 , 10]
#c3 : ax1=[0 , 10]  ax2= [-30 , -2]
#Right part of the plot
msk_1 = ((df_tsne['ax1']>10) & (df_tsne['ax1']<35 )) & ((df_tsne['ax2']>-20) &(df_tsne['ax2']<10))
cluster_1 = df_tsne[msk_1]
indexes_1 = cluster_1.index 
ex_1 = df_2016.loc[indexes_1]

#left part of the plot
msk_2 = ((df_tsne['ax1']>-30) & (df_tsne['ax1']<-5 )) & ((df_tsne['ax2']>-10) &(df_tsne['ax2']<10))
cluster_2 = df_tsne[msk_2]
indexes_2 = cluster_2.index 
ex_2 = df_2016.loc[indexes_2]    

#bottom part of the plot
msk_3 = ((df_tsne['ax1']>0) & (df_tsne['ax1']<10)) & ((df_tsne['ax2']>-30) &(df_tsne['ax2']<-2))
cluster_3 = df_tsne[msk_3]
indexes_3 = cluster_3.index 
ex_3 = df_2016.loc[indexes_3]

#top part of the plot
msk_4_1 = ((df_tsne['ax1']>-18) & (df_tsne['ax1']<-3)) & ((df_tsne['ax2']>18) &(df_tsne['ax2']<30))
msk_4_2 = ((df_tsne['ax1']>0) & (df_tsne['ax1']<3.5)) & ((df_tsne['ax2']>0) &(df_tsne['ax2']<13))
cluster_4 = df_tsne[msk_4_1 | msk_4_2]
indexes_4 = cluster_4.index 
ex_4 = df_2016.loc[indexes_4]

means_c1 = ex_1.mean(axis=0)
means_c2 = ex_2.mean(axis=0)
means_c3 = ex_3.mean(axis=0)
means_c4 = ex_4.mean(axis=0)

means = pd.DataFrame([means_c1,means_c2,means_c3,means_c4], ['c1','c2','c3','c4'])
means
ex_counties = pd.concat([ex_1.sample(1,random_state=17), #countie from c1 17 27
                         ex_2.sample(1,random_state=21), #countie from c2
                         ex_3.sample(1,random_state=33), #countie from c3
                         ex_4.sample(1,random_state=57)], axis=0)

ex_counties
# A function that will be used later to put the name of state and name of pollutant in the query

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text
states = ['California','Texas','Florida',
          'Mississippi','Vermont','Alaska'] 
pollutants = ['o3','co','so2','pm25_frm']

QUERY = """
    SELECT EXTRACT(YEAR FROM pollutant.date_local) as Year , AVG(pollutant.aqi) as AvgAQI_State
    FROM
      `bigquery-public-data.epa_historical_air_quality.pollutant_daily_summary` as pollutant
      WHERE pollutant.poc = 1 AND  pollutant.state_name = 'State'
    GROUP BY Year
    ORDER BY Year ASC
        """

dict_pol={}
for elem_g in pollutants : 
    dict_pol[elem_g] = None 
    for elem_s in states :
        dic = {"State": elem_s, "pollutant": elem_g}
        query = replace_all(QUERY, dic)
        temp = bq_assistant.query_to_pandas(query).set_index('Year')
        dict_pol[elem_g] = pd.concat([dict_pol[elem_g], temp], axis=1, join='inner')
        
dict_pol['co'].head(10)
fig, axs = plt.subplots(figsize=(20,12),ncols=2,nrows=2 )
dict_pol['o3'].plot( y=['AvgAQI_California','AvgAQI_Texas','AvgAQI_Florida',
                        'AvgAQI_Alaska','AvgAQI_Mississippi','AvgAQI_Vermont'], ax=axs[0,0],
                    title='Evolution of o3')
dict_pol['pm25_frm'].plot( y=['AvgAQI_California','AvgAQI_Texas','AvgAQI_Florida',
                              'AvgAQI_Alaska','AvgAQI_Mississippi','AvgAQI_Vermont'], ax=axs[0,1],
                          title='Evolution of pm2.5')
dict_pol['so2'].plot( y=['AvgAQI_California','AvgAQI_Texas','AvgAQI_Florida',
                         'AvgAQI_Alaska','AvgAQI_Mississippi','AvgAQI_Vermont'], ax=axs[1,0],
                     title='Evolution of so2')
dict_pol['co'].plot( y=['AvgAQI_California','AvgAQI_Texas','AvgAQI_Florida',
                        'AvgAQI_Alaska','AvgAQI_Mississippi','AvgAQI_Vermont'], ax=axs[1,1],
                   title='Evolution of co')


plt.show();
QUERYtemp = """
    SELECT
       EXTRACT(DAYOFYEAR FROM T.date_local) AS Day, AVG(T.arithmetic_mean) AS Temperature
    FROM
      `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary` as T
    WHERE
      T.state_name = 'California'
      AND EXTRACT(YEAR FROM T.date_local) = 2016
    GROUP BY Day
    ORDER BY Day
"""

QUERYrh = """
    SELECT
       EXTRACT(DAYOFYEAR FROM rh.date_local) AS Day, AVG(rh.arithmetic_mean) AS Humidity
    FROM
      `bigquery-public-data.epa_historical_air_quality.rh_and_dp_daily_summary` as rh
    WHERE
      rh.state_name = 'California'
      AND rh.parameter_name = 'Relative Humidity'
      AND EXTRACT(YEAR FROM rh.date_local) = 2016
    GROUP BY Day
    ORDER BY Day
"""

QUERYo3day = """
    SELECT
       EXTRACT(DAYOFYEAR FROM o3.date_local) AS Day, AVG(o3.aqi) AS o3_AQI
    FROM
      `bigquery-public-data.epa_historical_air_quality.o3_daily_summary` as o3
    WHERE
      o3.state_name = 'California'
      AND EXTRACT(YEAR FROM o3.date_local) = 2016
    GROUP BY Day
    ORDER BY Day
"""

QUERYpm25day = """
    SELECT
       EXTRACT(DAYOFYEAR FROM pm25.date_local) AS Day, AVG(pm25.aqi) AS pm25_AQI
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` as pm25
    WHERE
      pm25.state_name = 'California'
      AND pm25.sample_duration = '24 HOUR'
      AND EXTRACT(YEAR FROM pm25.date_local) = 2016
    GROUP BY Day
    ORDER BY Day
"""

df_temp = bq_assistant.query_to_pandas(QUERYtemp).set_index('Day')
df_pres = bq_assistant.query_to_pandas(QUERYrh).set_index('Day')
df_o3daily = bq_assistant.query_to_pandas(QUERYo3day).set_index('Day')
df_pm25daily = bq_assistant.query_to_pandas(QUERYpm25day).set_index('Day')

df_daily = pd.concat([df_temp, df_pres, df_o3daily, df_pm25daily], axis=1, join='inner')

df_daily.sample(10,random_state = 42)
corr = df_daily.corr()

# plot the heatmap
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=False, linewidths=.5, cbar_kws={"shrink": .5});
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
 
f, axs = plt.subplots(2,2,figsize=(15,10))
# multiple line plot
num=0
for column in df_daily :
    num+=1 # Find the right spot on the plot
    plt.subplot(2,2, num)
    # plot every groups, but discreet
    for v in df_daily : 
        plt.plot(df_daily.index, df_daily[v], marker='', color='grey', linewidth=0.6, alpha=0.3)
    # Plot the lineplot
    plt.plot(df_daily.index, df_daily[column], marker='',
             color=palette(num), linewidth=2.4, alpha=0.9, label=column)
    # Same limits for everybody!
    plt.xlim(0,370)
    plt.ylim(0,100)
    # Not ticks everywhere

    # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )

plt.suptitle("Temperature and Humidity impact on Ozone and Particulate Matter", fontsize=17, fontweight=0, color='black', style='italic', y=1.0);
 

print('The average o3 AQI between day #150 and day #250 is' , round(df_daily['o3_AQI'].iloc[150:250].mean(),2))
temp = list(range(0, 60)) + list(range(310,366))
print('The average PM2.5 AQI during late fall / early winter is' , round(df_daily['pm25_AQI'].iloc[temp].mean(),2))
bq_assistant_global = BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

QUERY_year = """
    SELECT
       EXTRACT(YEAR FROM globalAQ.timestamp) AS Year,unit, count(*) AS Total_measures
    FROM
      `bigquery-public-data.openaq.global_air_quality` as globalAQ
    GROUP BY Year , unit
    ORDER BY Total_measures
"""

df_year = bq_assistant_global.query_to_pandas(QUERY_year).set_index('Year')
df_year
QUERY_countries = """
    SELECT
       country, count(*) as nbmeasures, pollutant
    FROM
      `bigquery-public-data.openaq.global_air_quality` as globalAQ
    WHERE EXTRACT(YEAR FROM globalAQ.timestamp) = 2018 AND unit = 'µg/m³'
    GROUP BY country, pollutant
"""
df_countries = bq_assistant_global.query_to_pandas(QUERY_countries)

df_countries[df_countries['country']=='US']
countries = list(df_countries[(df_countries['pollutant']=='pm25') & (df_countries['nbmeasures']>30 )]['country'])
countries
QUERY_global = """
    SELECT
       country, AVG(value) as Concentration
    FROM
      `bigquery-public-data.openaq.global_air_quality` as globalAQ
    WHERE 
       EXTRACT(YEAR FROM globalAQ.timestamp) = 2018 
       AND unit = 'µg/m³'
       AND country IN ('US','SK','IN','NO','BE','CL','TW','CZ','ES','NL','AU','CA',
                       'CN','GB','DE', 'FR')
       AND pollutant = 'pm25'
       AND value > 0
    GROUP BY country
"""

df_global = bq_assistant_global.query_to_pandas(QUERY_global)

f,ax = plt.subplots(figsize=(14,6))
sns.barplot(df_global['country'],df_global['Concentration'])
plt.title('Concentration of PM2.5 in the air over the world');
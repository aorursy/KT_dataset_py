import pandas as pd
# Load pandas
import pandas as pd
# Load numpy
import numpy as np
# Set random seed
np.random.seed(0)
from google.cloud import bigquery
from bq_helper import BigQueryHelper
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
# google bigquery library for quering data
from google.cloud import bigquery
# BigQueryHelper for converting query result direct to dataframe
from bq_helper import BigQueryHelper
import matplotlib.pyplot as plt
import seaborn as sns
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pollutants = ['o3','co','no2','so2','pm25_frm','pm25_nonfrm']

QUERY2017 = """
    SELECT
        pollutant.state_name AS State, AVG(pollutant.aqi) AS AvgAQI_pollutant
    FROM
      `bigquery-public-data.epa_historical_air_quality.pollutant_daily_summary` as pollutant
    WHERE
      pollutant.poc = 1
      AND EXTRACT(YEAR FROM pollutant.date_local) = 2017
    GROUP BY 
      pollutant.state_name
"""

df_2017 = None
for elem_g in pollutants : 
    query = QUERY2017.replace("pollutant", elem_g)
    temp = bq_assistant.query_to_pandas(query).set_index('State')
    df_2017 = pd.concat([df_2017, temp], axis=1, join='outer')
df_2017=df_2017.apply(lambda x: x.fillna(x.mean()),axis=0)
df_2017['AvtTotal']=(df_2017['AvgAQI_o3']+df_2017['AvgAQI_co']+df_2017['AvgAQI_no2']+df_2017['AvgAQI_so2']+df_2017['AvgAQI_pm25_nonfrm'])/5
df_2017=df_2017.sort_values(by=['AvtTotal'], ascending=False)
df_2017.head(5)
df_2017_state=df_2017
df_2017_state['State'] = df_2017_state.index
df_2017_state=df_2017_state.sort_values(by=['AvtTotal'], ascending=True)
plt.subplots(figsize=(15,7))
sns.barplot(x='State', y='AvtTotal',data=df_2017_state,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Air Quality Index', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('States', fontsize=20)
plt.title('Statewise Total Average Pollution in USA in 2017 ', fontsize=24)
plt.show()
states = df_2017.head(5).index.tolist()
states
QUERY = """
    SELECT EXTRACT(YEAR FROM pollutant.date_local) as Year , AVG(pollutant.aqi) as AvgAQI_State
    FROM
      `bigquery-public-data.epa_historical_air_quality.pollutant_daily_summary` as pollutant
      WHERE pollutant.poc = 1 AND  pollutant.state_name = 'State'
    GROUP BY Year
    ORDER BY Year ASC
        """
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

dict_pol={}
for elem_g in pollutants : 
    dict_pol[elem_g] = None 
    for elem_s in states :
        dic = {"State": elem_s, "pollutant": elem_g}
        query = replace_all(QUERY, dic)
        temp = bq_assistant.query_to_pandas(query).set_index('Year')
        dict_pol[elem_g] = pd.concat([dict_pol[elem_g], temp], axis=1, join='inner')
import matplotlib.pyplot as plt
fig, axs = plt.subplots(figsize=(20,28),ncols=2,nrows=3 )
dict_pol['o3'].plot( y=['AvgAQI_Hawaii','AvgAQI_Arizona','AvgAQI_Illinois',
                        'AvgAQI_Oklahoma','AvgAQI_California'], ax=axs[0,0],
                    title='Evolution of o3')
dict_pol['co'].plot( y=['AvgAQI_Hawaii','AvgAQI_Arizona','AvgAQI_Illinois',
                        'AvgAQI_Oklahoma','AvgAQI_California'], ax=axs[0,1],
                    title='Evolution of co')
dict_pol['no2'].plot( y=['AvgAQI_Hawaii','AvgAQI_Arizona','AvgAQI_Illinois',
                        'AvgAQI_Oklahoma','AvgAQI_California'], ax=axs[1,0],
                    title='Evolution of no2')
dict_pol['so2'].plot( y=['AvgAQI_Hawaii','AvgAQI_Arizona','AvgAQI_Illinois',
                        'AvgAQI_Oklahoma','AvgAQI_California'], ax=axs[1,1],
                    title='Evolution of so2')
dict_pol['pm25_frm'].plot( y=['AvgAQI_Hawaii','AvgAQI_Arizona','AvgAQI_Illinois',
                        'AvgAQI_Oklahoma','AvgAQI_California'], ax=axs[2,0],
                    title='Evolution of pm25_frm')
dict_pol['pm25_nonfrm'].plot( y=['AvgAQI_Hawaii','AvgAQI_Arizona','AvgAQI_Illinois',
                        'AvgAQI_Oklahoma','AvgAQI_California'], ax=axs[2,1],
                    title='Evolution of pm25_nonfrm')

plt.show();
pm25QUERY = """
    SELECT
        extract(DAYOFYEAR from date_local) as day_of_year, aqi AS aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      poc = 1
      AND state_name IN ('Hawaii','Arizona','Illinois','Oklahoma','California')
      AND sample_duration = "24 HOUR"
      AND EXTRACT(YEAR FROM date_local) = 2017
    ORDER BY 
    day_of_year
"""
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
df_pm25 = bq_assistant.query_to_pandas(pm25QUERY)
df_pm25.plot(x='day_of_year', y='aqi', style='.');
# aqi
weather_query = """SELECT AVG(arithmetic_mean) as `AverageTemperature`,state_name as `State`
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        GROUP BY State
        ORDER BY AverageTemperature DESC
        """
df_weather_all_states = bq_assistant.query_to_pandas_safe(weather_query,max_gb_scanned=2)
plt.figure(figsize=(14,15))
sns.barplot(df_weather_all_states['AverageTemperature'], df_weather_all_states['State'], palette='gist_rainbow')
plt.title("Statewise Average Temperature in USA 1990-2017");
# aqi
weather_query = """SELECT AVG(arithmetic_mean) as `AverageTemperature`,state_name as `State`
        FROM `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary`
        WHERE
        EXTRACT(YEAR FROM date_local) = 2017
        GROUP BY State
        ORDER BY AverageTemperature DESC
        """
df_weather_all_states_2017 = bq_assistant.query_to_pandas_safe(weather_query,max_gb_scanned=2)
plt.figure(figsize=(14,15))
sns.barplot(df_weather_all_states_2017['AverageTemperature'], df_weather_all_states_2017['State'], palette='gist_rainbow')
plt.title("Statewise Average Temperature in USA 2017");
df_2017_with_temp = df_2017
df_2017_with_temp['Avg_temp'] = df_2017_with_temp['State'].map(df_weather_all_states_2017.set_index('State')['AverageTemperature'])
df_2017_with_temp = df_2017_with_temp[['State', 'AvgAQI_o3', 'AvgAQI_co', 'AvgAQI_no2', 'AvgAQI_so2', 'AvgAQI_pm25_frm', 'AvgAQI_pm25_nonfrm', 'AvtTotal', 'Avg_temp']]
df_2017_with_temp.head(10)
fig = plt.figure(figsize=(12, 6))
pm25 = fig.add_subplot(121)

pm25.hist(df_2017_with_temp.AvgAQI_pm25_nonfrm, bins=80)
pm25.set_xlabel('micrograms per cubic meter')
pm25.set_title("Histogram of PM2.5")

plt.show()
QUERYtemp = """
    SELECT
       EXTRACT(DAYOFYEAR FROM T.date_local) AS Day, AVG(T.arithmetic_mean) AS Temperature
    FROM
      `bigquery-public-data.epa_historical_air_quality.temperature_daily_summary` as T
    WHERE
      T.state_name IN ('Hawaii', 'Arizona', 'Illinois', 'Oklahoma', 'California')

    GROUP BY Day
    ORDER BY Day
"""

QUERYhumid = """
    SELECT
       EXTRACT(DAYOFYEAR FROM rh.date_local) AS Day, AVG(rh.arithmetic_mean) AS Humidity
    FROM
      `bigquery-public-data.epa_historical_air_quality.rh_and_dp_daily_summary` as rh
    WHERE
      rh.state_name IN ('Hawaii', 'Arizona', 'Illinois', 'Oklahoma', 'California')
      AND rh.parameter_name = 'Relative Humidity'

    GROUP BY Day
    ORDER BY Day
"""

QUERYo3day = """
    SELECT
       EXTRACT(DAYOFYEAR FROM o3.date_local) AS Day, AVG(o3.aqi) AS o3_AQI
    FROM
      `bigquery-public-data.epa_historical_air_quality.o3_daily_summary` as o3
    WHERE
      o3.state_name IN ('Hawaii', 'Arizona', 'Illinois', 'Oklahoma', 'California')
  
    GROUP BY Day
    ORDER BY Day
"""

QUERYno2day = """
    SELECT
       EXTRACT(DAYOFYEAR FROM no2.date_local) AS Day, AVG(no2.aqi) AS no2_AQI
    FROM
      `bigquery-public-data.epa_historical_air_quality.no2_daily_summary` as no2
    WHERE
      no2.state_name IN ('Hawaii', 'Arizona', 'Illinois', 'Oklahoma', 'California')

    GROUP BY Day
    ORDER BY Day
"""

QUERYcoday = """
    SELECT
       EXTRACT(DAYOFYEAR FROM co.date_local) AS Day, AVG(co.aqi) AS co_AQI
    FROM
      `bigquery-public-data.epa_historical_air_quality.co_daily_summary` as co
    WHERE
      co.state_name IN ('Hawaii', 'Arizona', 'Illinois', 'Oklahoma', 'California')

    GROUP BY Day
    ORDER BY Day
"""

QUERYpm25day = """
    SELECT
       EXTRACT(DAYOFYEAR FROM pm25.date_local) AS Day, AVG(pm25.aqi) AS pm25_AQI
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` as pm25
    WHERE
      pm25.state_name IN ('Hawaii', 'Arizona', 'Illinois', 'Oklahoma', 'California')
      AND pm25.sample_duration = '24 HOUR'

    GROUP BY Day
    ORDER BY Day
"""

df_temp = bq_assistant.query_to_pandas(QUERYtemp).set_index('Day')
df_humid = bq_assistant.query_to_pandas(QUERYhumid).set_index('Day')
df_o3daily = bq_assistant.query_to_pandas(QUERYo3day).set_index('Day')
df_no2daily = bq_assistant.query_to_pandas(QUERYno2day).set_index('Day')
df_codaily = bq_assistant.query_to_pandas(QUERYcoday).set_index('Day')
df_pm25daily = bq_assistant.query_to_pandas(QUERYpm25day).set_index('Day')

df_daily_regr = pd.concat([df_o3daily, df_pm25daily, df_no2daily, df_codaily, df_humid, df_temp], axis=1, join='inner')

df_daily_regr.sample(10,random_state = 42)
import seaborn as sns
%matplotlib inline
corr = df_daily_regr.corr()
# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
import statsmodels.api as sm
from statsmodels.formula.api import ols
ds = df_daily_regr
ds.head()
m = ols('Temperature ~ o3_AQI + no2_AQI + co_AQI + pm25_AQI + Humidity',ds).fit()
print (m.summary())
plt.scatter(ds['pm25_AQI'], ds['Temperature'])
plt.xlabel('Concentration of PM2.5')
plt.ylabel('Temperature')
plt.show()
plt.scatter(ds['co_AQI'], ds['Temperature'])
plt.xlabel('Concentration of CO')
plt.ylabel('Temperature')
plt.show()
plt.scatter(ds['no2_AQI'], ds['Temperature'])
plt.xlabel('Concentration of NO2')
plt.ylabel('Temperature')
plt.show()
plt.scatter(ds['o3_AQI'], ds['Temperature'])
plt.xlabel('Concentration of Ozone')
plt.ylabel('Temperature')
plt.show()
plt.scatter(ds['Humidity'], ds['Temperature'])
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn import linear_model

ds = df_daily_regr
ds = ds.drop('Humidity', axis=1)
y = ds['Temperature']
ds.head()
X = ds
X = X.drop('Temperature', axis=1)
X.head()
lm = linear_model.LinearRegression()
model = lm.fit(X,y)
print(lm.score(X,y))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
ds = df_daily_regr

# View the top 5 rows
ds.head()
print('Number of observations in the data:', len(ds))

# labels - predict
labels=np.array(ds['Temperature'])
# View features
ds=ds.drop('Temperature',axis=1)

feature_list=list(ds.columns)
ds=np.array(ds)

train_features, test_features, train_labels, test_labels = train_test_split(ds, labels, test_size=0.25, random_state=42)

#print("Training features shape: ", train_features.shape)
#print("Test features shape: ", test_features.shape)
#print("Training labels shape: ", train_labels.shape)
#print("Test labels shape: ", test_labels.shape)

rf = RandomForestRegressor(n_estimators=1000, random_state=0)

rf.fit(train_features, train_labels)
predictions=rf.predict(test_features)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# Build a plot
plt.scatter(predictions, test_labels)
plt.xlabel('Prediction')
plt.ylabel('Real value')

# Now add the perfect prediction line
diagonal = np.linspace(0, np.max(test_labels), 100)
plt.plot(diagonal, diagonal, '-r')
plt.show()

from sklearn.tree import export_graphviz

# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt
%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
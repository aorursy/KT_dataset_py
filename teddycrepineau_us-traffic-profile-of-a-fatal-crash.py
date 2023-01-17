import bq_helper as bqh #Import BigQueryHelper
import pandas as pd #Import Pandas
import matplotlib.pyplot as plt #Import Matplotlib
import numpy as np #Import numpy
from scipy.stats import pearsonr
us_traffic_fat = bqh.BigQueryHelper(active_project = 'bigquery-public-data',
                                   dataset_name = 'nhtsa_traffic_fatalities')

query = """
        SELECT
            state_name,
            CASE
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 0 THEN '12AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 1 THEN '1AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 2 THEN '2AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 3 THEN '3AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 4 THEN '4AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 5 THEN '5AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 6 THEN '6AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 7 THEN '7AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 8 THEN '8AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 9 THEN '9AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 10 THEN '10AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 11 THEN '11AM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 12 THEN '12PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 13 THEN '1PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 14 THEN '2PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 15 THEN '3PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 16 THEN '4PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 17 THEN '5PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 18 THEN '6PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 19 THEN '7PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 20 THEN '8PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 21 THEN '9PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 22 THEN '10PM'
                WHEN EXTRACT(HOUR FROM timestamp_of_crash) = 23 THEN '11PM'
                END Hour_Of_Day,
            CASE
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 1 THEN 'Sunday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 2 THEN 'Monday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 3 THEN 'Tuesday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 4 THEN 'Wednesday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 5 THEN 'Thursday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 6 THEN 'Friday'
                WHEN EXTRACT(DAYOFWEEK FROM timestamp_of_crash) = 7 THEN 'Saturday'
            END Day_Of_Week,
            CASE
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 1 THEN 'January'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 2 THEN 'February'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 3 THEN 'March'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 4 THEN 'April'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 5 THEN 'May'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 6 THEN 'June'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 7 THEN 'July'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 8 THEN 'August'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 9 THEN 'September'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 10 THEN 'October'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 11 THEN 'November'
                WHEN EXTRACT(MONTH FROM timestamp_of_crash) = 12 THEN 'December'
            END Month,
            functional_system_name AS Trafficway_Type,
            type_of_intersection AS Intersection,
            light_condition_name AS Light_Condition,
            atmospheric_conditions_1_name AS Atmospheric_conditions,
            COUNT(DISTINCT consecutive_number) AS num_fatalities
        FROM
            `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        GROUP BY 1,2,3,4,5,6,7,8
        ORDER BY state_name
        """

#Put query in a DF, join it with 2016_state_population file and add calculated field
df = us_traffic_fat.query_to_pandas(query)
df_state_population = pd.read_csv("../input/2016_states_population.csv")

df[:10]
df_fat_per_states = df.loc[:,['state_name','num_fatalities']]
df_fat_per_states = df_fat_per_states.groupby(['state_name'])['num_fatalities'].sum().reset_index()

df_join = pd.merge(df_state_population, df_fat_per_states, how='inner', on='state_name')
df_join['crashes_per_thousand'] = df_join['num_fatalities'] / df_join['2016_population'] * 1000 #Calculate fatal carsh per 1000 inhabitants
df_join = df_join.sort_values(by=['crashes_per_thousand'], ascending=False) #Sort data by 'crashes_per_thousands'

#Define figure size, add label to x axis and plot graph

state_name_list = pd.Series.tolist(df_join['state_name'])
y_pos = np.arange(len(state_name_list))
plt.figure(figsize=(14,5))
plt.title('Fatalities Per Thousand Inhabitants Per State')
plt.ylabel('Fatalities Per Thousand')
plt.xlabel('States')
plt.xticks(y_pos, state_name_list, rotation='vertical')
plt.bar(y_pos, df_join['crashes_per_thousand'], align='center', alpha=0.5)
plt.show()
total_fat = df['num_fatalities'].sum()
df_fat_per_hour = df.loc[:,['Hour_Of_Day','num_fatalities']]
df_fat_per_hour = df_fat_per_hour.groupby(['Hour_Of_Day'])['num_fatalities'].sum().reset_index()
df_fat_per_hour['fat_per_h_perc'] = df_fat_per_hour['num_fatalities'] / total_fat * 100

df_fat_per_hour = df_fat_per_hour.sort_values(by=['fat_per_h_perc'],ascending=False)

hod_list = pd.Series.tolist(df_fat_per_hour['Hour_Of_Day'])
hod_list_x_label = pd.Series.tolist(df_fat_per_hour['Hour_Of_Day'])
y_pos = np.arange(len(hod_list))
plt.figure(figsize=(14,5))
plt.xticks(y_pos, hod_list_x_label)
plt.xlabel('Hour Of Day')
plt.ylabel('Percentage of Fatalities')
plt.title('Fatalities Per Hour (National)')
plt.bar(y_pos, df_fat_per_hour['fat_per_h_perc'],align='center', alpha=0.5)
plt.show()
total_fat = df['num_fatalities'].sum()
df_fat_per_dow = df.loc[:,['Day_Of_Week','num_fatalities']]
df_fat_per_dow = df_fat_per_dow.groupby(['Day_Of_Week'])['num_fatalities'].sum().reset_index()
df_fat_per_dow['fat_per_dow_perc'] = df_fat_per_dow['num_fatalities'] / total_fat * 100

df_fat_per_dow = df_fat_per_dow.sort_values(by=['fat_per_dow_perc'], ascending=False)

dow_list = pd.Series.tolist(df_fat_per_dow['Day_Of_Week'])
dow_list_x_label = pd.Series.tolist(df_fat_per_dow['Day_Of_Week'])
y_pos = np.arange(len(dow_list))
plt.figure(figsize=(14,5))
plt.xticks(y_pos, dow_list_x_label)
plt.xlabel('Day Of Week')
plt.ylabel('Percentage of Fatalities')
plt.title('Fatalities Per Day Of Week (National)')
plt.bar(y_pos, df_fat_per_dow['fat_per_dow_perc'],align='center', alpha=0.5)
plt.show()
total_fat = df['num_fatalities'].sum()
df_fat_per_moy = df.loc[:,['Month','num_fatalities']]
df_fat_per_moy = df_fat_per_moy.groupby(['Month'])['num_fatalities'].sum().reset_index()
df_fat_per_moy['fat_per_moy_perc'] = df_fat_per_moy['num_fatalities'] / total_fat * 100

df_fat_per_moy = df_fat_per_moy.sort_values(by=['fat_per_moy_perc'], ascending=False)

moy_list = pd.Series.tolist(df_fat_per_moy['Month'])
moy_list_x_label = pd.Series.tolist(df_fat_per_moy['Month'])
y_pos = np.arange(len(moy_list))
plt.figure(figsize=(14,5))
plt.xticks(y_pos, moy_list_x_label)
plt.xlabel('Month Of Year')
plt.ylabel('Percentage of Fatalities')
plt.title('Fatalities Per Month (National)')
plt.bar(y_pos, df_fat_per_moy['fat_per_moy_perc'],align='center', alpha=0.5)
plt.show()
total_fat = df['num_fatalities'].sum()
df_fat_traffic = df.loc[:,['Trafficway_Type','num_fatalities']]
df_fat_traffic = df_fat_traffic.groupby(['Trafficway_Type'])['num_fatalities'].sum().reset_index()
df_fat_traffic['fat_per_traffic_perc'] = df_fat_traffic['num_fatalities'] / total_fat * 100

df_fat_traffic = df_fat_traffic.sort_values(by=['fat_per_traffic_perc'], ascending=False)

traffic_list = pd.Series.tolist(df_fat_traffic['Trafficway_Type'])
traffic_list_x_label = pd.Series.tolist(df_fat_traffic['Trafficway_Type'])
y_pos = np.arange(len(traffic_list))
plt.figure(figsize=(14,5))
plt.xticks(y_pos, traffic_list_x_label, rotation='vertical')
plt.xlabel('Month Of Year')
plt.ylabel('Percentage of Fatalities')
plt.title('Fatalities Per Trafficway Type (National)')
plt.bar(y_pos, df_fat_traffic['fat_per_traffic_perc'],align='center', alpha=0.5)
plt.show()
total_fat = df['num_fatalities'].sum()
df_fat_atcon = df.loc[:,['Atmospheric_conditions','num_fatalities']]
df_fat_atcon = df_fat_atcon.groupby(['Atmospheric_conditions'])['num_fatalities'].sum().reset_index()
df_fat_atcon['fat_per_atcon_perc'] = df_fat_atcon['num_fatalities'] / total_fat * 100

df_fat_atcon = df_fat_atcon.sort_values(by=['fat_per_atcon_perc'], ascending=False)

atcon_list = pd.Series.tolist(df_fat_atcon['Atmospheric_conditions'])
atcon_list_x_label = pd.Series.tolist(df_fat_atcon['Atmospheric_conditions'])
size = pd.Series.tolist(df_fat_atcon['fat_per_atcon_perc'])
y_pos = np.arange(len(atcon_list))
plt.figure(figsize=(6,6))
plt.xticks(y_pos, atcon_list_x_label, rotation='vertical')
plt.title('Fatalities Per Atmospheric Condition')
plt.pie(df_fat_atcon['fat_per_atcon_perc'])
plt.legend(['%s, %1.1f %%' % (l, s) for l, s in zip(atcon_list_x_label,size)], loc='best')
plt.show()
import bq_helper as bqh
import pandas as pd

us_traffic_fat_per_state = bqh.BigQueryHelper(active_project = 'bigquery-public-data',
                                            dataset_name = 'nhtsa_traffic_fatalities')

query = """
        SELECT
            state_name,
            COUNT(DISTINCT consecutive_number) AS fatalities
        FROM
            `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        GROUP BY
            1
        ORDER BY
            fatalities DESC
        """

df_clear_days_2016 = pd.read_csv("../input/2016_clear_days.csv")
df_clear_days_2016['clear_day_%'] = df_clear_days_2016['CL'] / 365 * 100
df_clear_days_2016 = df_clear_days_2016.rename(index=str, columns={'State_Name':'state_name'})
state_fatalities = us_traffic_fat_per_state.query_to_pandas(query)

average_clear_day = df_clear_days_2016['CL'].sum()/df_clear_days_2016['CL'].count()
df_join_w = pd.merge(state_fatalities, df_clear_days_2016, how='inner', on='state_name')
df_join_w['fatalities_%'] = df_join_w['fatalities'] / df_join_w['fatalities'].sum() * 100
df_join_w = df_join_w[df_join_w['CL'] > average_clear_day].reset_index(drop=True)
df_join_w.loc['Total'] = df_join_w.sum(numeric_only=True)
df_join_w
total_fat = df['num_fatalities'].sum()
df_fat_light = df.loc[:,['Light_Condition','num_fatalities']]
df_fat_light = df_fat_light.groupby(['Light_Condition'])['num_fatalities'].sum().reset_index()

df_fat_light['fat_per_light_perc'] = (df_fat_light['num_fatalities'] / total_fat) * 100

df_fat_light = df_fat_light.sort_values(by=['fat_per_light_perc'], ascending=False)

light_list = pd.Series.tolist(df_fat_light['Light_Condition'])
light_list_x_label = pd.Series.tolist(df_fat_light['Light_Condition'])
size = pd.Series.tolist(df_fat_light['fat_per_light_perc'])
y_pos = np.arange(len(light_list))
plt.figure(figsize=(6,6))
plt.xticks(y_pos, light_list_x_label, rotation='vertical')
plt.title('Fatalities Per Light Condition')
plt.pie(df_fat_light['fat_per_light_perc'])
plt.legend(['%s, %1.1f %%' % (l, s) for l, s in zip(light_list_x_label,size)], loc='best')
plt.show()
total_fat = df['num_fatalities'].sum()
df_fat_inter = df.loc[:,['Intersection','num_fatalities']]
df_fat_inter = df_fat_inter.groupby(['Intersection'])['num_fatalities'].sum().reset_index()

df_fat_inter['fat_per_inter_perc'] = (df_fat_inter['num_fatalities'] / total_fat) * 100

df_fat_inter = df_fat_inter.sort_values(by=['fat_per_inter_perc'], ascending=False)

inter_list = pd.Series.tolist(df_fat_inter['Intersection'])
inter_list_x_label = pd.Series.tolist(df_fat_inter['Intersection'])
size = pd.Series.tolist(df_fat_inter['fat_per_inter_perc'])
y_pos = np.arange(len(inter_list))
plt.figure(figsize=(6,6))
plt.xticks(y_pos, inter_list_x_label, rotation='vertical')
plt.title('Fatalities Per Intersection Type')
plt.pie(df_fat_inter['fat_per_inter_perc'])
plt.legend(['%s, %1.1f %%' % (l, s) for l, s in zip(inter_list_x_label,size)], loc='best')
plt.show()
df_fatalities = df.loc[:,['state_name','num_fatalities']]
df_fatalities = df_fatalities.groupby(['state_name'])['num_fatalities'].sum().reset_index()
df_vehicle_mile = pd.read_csv('../input/2016_vehicle_mile.csv')

df_join = pd.merge(df_fatalities, df_vehicle_mile, how='inner', on='state_name')
df_join = pd.merge(df_join, df_state_population, how='inner', on='state_name')
df_join['crashes_per_thousand'] = df_join['num_fatalities'] / df_join['2016_population'] * 1000 #Calculate fatal carsh per 1000 inhabitants
df_join['vehicle_mile_per_thousand'] = df_join['vehicle-mile(millions)'] * 1000 / df_join['2016_population'] * 1000

df_target_state = df_join.iloc[[24,0,17,40,31,32,30,8,39,21]].reset_index(drop=True)

plt.figure(figsize=(6,6))

color = ['#CD6155','#808B96','#9B59B6','#99A3A4','#7D6608','#F39C12','#2980B9','#16A085','#ABEBC6','#AED6F1']
state_name = df_target_state['state_name'].tolist()
vehicle_mile = df_target_state['vehicle_mile_per_thousand']
fatalities = df_target_state['crashes_per_thousand']


for i in range(len(df_target_state['state_name'])):
    x = vehicle_mile[i]
    y = fatalities[i]
    plt.scatter(x,y, label=state_name[i])

plt.xlabel('vehicle-mile per thousand inhabitant')
plt.ylabel('fatalities per thousand inhabitant')
plt.legend()
plt.show()
var = pearsonr(vehicle_mile, fatalities)
print('Correlation Coefficient: ',var[0],'\n','P-value: ',var[1])
query_2 = """
            SELECT
                  state_name,
                  COUNT(DISTINCT consecutive_number) AS Impaired_Fatality,
                  SUM(number_of_drunk_drivers) AS Drunk_Drivers
            FROM
                `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            WHERE
                number_of_drunk_drivers >= 1
            GROUP BY
                1
        """


df_impaired = us_traffic_fat.query_to_pandas(query_2)

df_fatalities = df.loc[:,['state_name','num_fatalities']]
df_fatalities = df_fatalities.groupby(['state_name'])['num_fatalities'].sum().reset_index()


df_join = pd.merge(df_fatalities, df_impaired, how='inner',on='state_name')
df_join = pd.merge(df_join, df_state_population, how='inner', on='state_name')

df_join['impaired_fat_thousand'] = df_join['Impaired_Fatality'] / df_join['2016_population'] * 1000
df_join['fat_thousand'] = df_join['num_fatalities'] / df_join['2016_population'] * 1000

df_impaired_fat = df_join

df_impaired_fat.iloc[:10]
df_target_state_i = df_impaired_fat.iloc[[24,0,17,40,31,32,30,8,39,21]].reset_index(drop=True)

color = ['#CD6155','#808B96','#9B59B6','#99A3A4','#7D6608','#F39C12','#2980B9','#16A085','#ABEBC6','#AED6F1']
state_name = df_target_state_i['state_name'].tolist()
impaired_fat = df_target_state_i['impaired_fat_thousand']
fatalities = df_target_state_i['fat_thousand']


for i in range(len(df_target_state_i['state_name'])):
    x = impaired_fat[i]
    y = fatalities[i]
    plt.scatter(x,y, label=state_name[i])

plt.xlabel('Impaired Fatalities per thousand inhabitant')
plt.ylabel('fatalities per thousand inhabitant')
plt.legend()
plt.show()
var = pearsonr(impaired_fat, fatalities)
print('Correlation Coefficient: ',var[0],'\n','P-value: ',var[1])
urban_areas = pd.read_csv('../input/2010_census_urban_area.csv')
urban_areas['urbanization_index'] = (urban_areas['urban_areas'] / urban_areas['urban_areas'].mean() * 100).round(decimals=0)

urban_areas.iloc[:10]
vehicle_mile = pd.read_csv('../input/2016_vehicle_mile.csv')

df_join_uv = pd.merge(urban_areas, vehicle_mile, how='inner', on='state_name')
df_join_uvp = pd.merge(df_join_uv, df_state_population, how='inner', on='state_name')

df_fatalities = df.loc[:,['state_name','num_fatalities']]
df_fatalities = df_fatalities.groupby(['state_name'])['num_fatalities'].sum().reset_index()

df_join_uvpf = pd.merge(df_join_uvp, df_fatalities, how='inner', on='state_name')
df_join_uvpf['vehicle_mile_thousand'] = df_join_uvpf['vehicle-mile(millions)'] * 1000 / df_join_uvpf['2016_population'] * 1000
df_join_uvpf['fat_per_thousand'] = df_join_uvpf['num_fatalities'] / df_join_uvpf['2016_population'] * 1000


df_join_uvpf.iloc[:10]
df_target_state_f = df_join_uvpf.iloc[[1,39,24,31,16,33,38,18,30]].reset_index(drop=True)

color = ['#CD6155','#808B96','#9B59B6','#99A3A4','#7D6608','#F39C12','#2980B9','#16A085','#ABEBC6','#AED6F1']
state_name = df_target_state_f['state_name'].tolist()
urbanization = df_target_state_f['urbanization_index']
fatalities = df_target_state_f['fat_per_thousand']


for i in range(len(df_target_state_f['state_name'])):
    x = urbanization[i]
    y = fatalities[i]
    plt.scatter(x,y, label=state_name[i])

plt.xlabel('urbanization Index per thousand inhabitant (micro/metro)')
plt.ylabel('Fatalities Per Thousand')
plt.legend()
plt.show()
var = pearsonr(urbanization, fatalities)
print('Correlation Coefficient: ',var[0],'\n','P-value: ',var[1])
df_target_state_u = df_join_uvpf.iloc[[1,39,24,31,16,33,38,18,30]].reset_index(drop=True)

color = ['#CD6155','#808B96','#9B59B6','#99A3A4','#7D6608','#F39C12','#2980B9','#16A085','#ABEBC6','#AED6F1']
state_name = df_target_state_u['state_name'].tolist()
urbanization = df_target_state_u['urbanization_index']
vehicle_mile = df_target_state_u['vehicle_mile_thousand']


for i in range(len(df_target_state_u['state_name'])):
    x = urbanization[i]
    y = vehicle_mile[i]
    plt.scatter(x,y, label=state_name[i])

plt.xlabel('urbanization Index per thousand inhabitant (micro/metro)')
plt.ylabel('vehicle-mile per thousand inhabitant')
plt.legend()
plt.show()
var = pearsonr(urbanization, vehicle_mile)
print('Correlation Coefficient: ',var[0],'\n','P-value: ',var[1])
metro_areas = pd.read_csv("../input/2010_census_metro_area.csv")
metro_areas['urbanization_index'] = (metro_areas['metropolitan_areas'] / metro_areas['metropolitan_areas'].mean() * 100).round(decimals=0)

vehicle_mile = pd.read_csv('../input/2016_vehicle_mile.csv')

df_join_mv = pd.merge(metro_areas, vehicle_mile, how='inner', on='state_name')
df_join_mvp = pd.merge(df_join_mv, df_state_population, how='inner', on='state_name')

df_fatalities = df.loc[:,['state_name','num_fatalities']]
df_fatalities = df_fatalities.groupby(['state_name'])['num_fatalities'].sum().reset_index()

df_join_mvpf = pd.merge(df_join_mvp, df_fatalities, how='inner', on='state_name')
df_join_mvpf['vehicle_mile_thousand'] = df_join_mvpf['vehicle-mile(millions)'] * 1000 / df_join_mvpf['2016_population'] * 1000
df_join_mvpf['fat_per_thousand'] = df_join_mvpf['num_fatalities'] / df_join_mvpf['2016_population'] * 1000


df_join_mvpf.iloc[:10]
df_target_state_m = df_join_mvpf.iloc[[1,39,24,31,16,33,38,18,30]].reset_index(drop=True)

color = ['#CD6155','#808B96','#9B59B6','#99A3A4','#7D6608','#F39C12','#2980B9','#16A085','#ABEBC6','#AED6F1']
state_name = df_target_state_m['state_name'].tolist()
urbanization = df_target_state_m['urbanization_index']
vehicle_mile = df_target_state_m['vehicle_mile_thousand']


for i in range(len(df_target_state_m['state_name'])):
    x = urbanization[i]
    y = vehicle_mile[i]
    plt.scatter(x,y, label=state_name[i])

plt.xlabel('urbanization Index per thousand inhabitant (metro)')
plt.ylabel('vehicle-mile per thousand inhabitant')
plt.legend()
plt.show()
var = pearsonr(urbanization, vehicle_mile)
print('Correlation Coefficient: ',var[0],'\n','P-value: ',var[1])
import bq_helper
#Load in the dataset
us_traffic = bq_helper.BigQueryHelper(active_project="bigquery-public-data",\
                                      dataset_name="nhtsa_traffic_fatalities")
us_traffic.list_tables()
query_1 = """WITH accidents_2015_2016 as (SELECT consecutive_number, timestamp_of_crash
                                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                                        UNION ALL
                                        SELECT consecutive_number, timestamp_of_crash
                                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`)
                SELECT EXTRACT(HOUR from accidents_2015_2016.timestamp_of_crash) as crash_hour, COUNT(accidents_2015_2016.consecutive_number) as total_crashes
                FROM accidents_2015_2016
                GROUP BY crash_hour
                ORDER BY COUNT(accidents_2015_2016.consecutive_number) DESC
            """
us_traffic.estimate_query_size(query_1)
crashes_per_hour = us_traffic.query_to_pandas_safe(query_1)
crashes_per_hour.to_csv('total_crashes_per_hour.csv')
crashes_per_hour
from ggplot import *
ggplot(crashes_per_hour, aes(x='crash_hour', y='total_crashes')) +\
       geom_line() + xlab("Hour") + ylab("Number of Crashes") +\
    ggtitle("Crashes per Hour in 2015-2016")
query_1_extra = """WITH accidents_2015_2016 as (SELECT consecutive_number, timestamp_of_crash
                                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                                        UNION ALL
                                        SELECT consecutive_number, timestamp_of_crash
                                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`)
                SELECT EXTRACT(YEAR from timestamp_of_crash) as crash_year, EXTRACT(HOUR from timestamp_of_crash) as crash_hour, COUNT(consecutive_number) as total_crashes
                FROM accidents_2015_2016
                GROUP BY crash_year, crash_hour
                ORDER BY COUNT(consecutive_number) DESC
            """
us_traffic.estimate_query_size(query_1_extra)
crashes_per_hour_per_year = us_traffic.query_to_pandas_safe(query_1_extra)
crashes_per_hour_per_year.head()
ggplot(crashes_per_hour_per_year, aes(x='crash_hour', y='total_crashes')) +\
       geom_line() + xlab("Hour") + ylab("Number of Crashes") +\
    ggtitle("Crashes per Hour in 2015-2016") + facet_wrap("crash_year")
query_1_extra2 = """WITH accidents_2015_2016 as (SELECT consecutive_number, timestamp_of_crash
                                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                                        UNION ALL
                                        SELECT consecutive_number, timestamp_of_crash
                                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`)
                SELECT EXTRACT(YEAR from timestamp_of_crash) as crash_year,
                    EXTRACT(DAYOFWEEK from timestamp_of_crash) as crash_dow,
                    EXTRACT(HOUR from timestamp_of_crash) as crash_hour,
                        COUNT(consecutive_number) as total_crashes
                FROM accidents_2015_2016
                GROUP BY crash_year, crash_dow, crash_hour
                ORDER BY COUNT(consecutive_number) DESC
            """
crashes_year_dow_hour = us_traffic.query_to_pandas_safe(query_1_extra2)
crashes_year_dow_hour.to_csv('total_crashes_year_dow_hour.csv')
crashes_year_dow_hour.head()
data = crashes_year_dow_hour
data_max = crashes_year_dow_hour['total_crashes'].max()

def facet(data,color,**kws):
    data = data.pivot(index='crash_dow',columns='crash_hour',values='total_crashes')
    g = sns.heatmap(data, cmap='Blues', **kws)

    
with sns.plotting_context(font_scale=5.5):    
    g = sns.FacetGrid(data, col="crash_year")

cbar_ax=g.fig.add_axes([0.92,0.3,0.02,0.4])
    
g = g.map_dataframe(facet, cbar_ax=cbar_ax, vmin=0, vmax=data_max)
g.add_legend()
g.set_titles(col_template="{col_name}", fontweight='bold', fontsize=18)
g.fig.subplots_adjust(right=0.9)
query_2 = """ WITH vehicle_2015_2016 as (SELECT registration_state_name, hit_and_run
                                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                                        UNION ALL
                                        SELECT registration_state_name, hit_and_run
                                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`)
            SELECT registration_state_name, COUNT(hit_and_run) as total_hit_and_runs
            FROM vehicle_2015_2016
            WHERE hit_and_run='Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC;
            """
us_traffic.estimate_query_size(query_2)
answer_2 = us_traffic.query_to_pandas_safe(query_2)
answer_2.to_csv('states_by_total_hit_and_runs.csv')
answer_2

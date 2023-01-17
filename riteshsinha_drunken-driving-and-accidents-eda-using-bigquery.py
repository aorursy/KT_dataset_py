from google.cloud import bigquery

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

import bq_helper

from tabulate import tabulate

client = bigquery.Client()



fars_data = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "nhtsa_traffic_fatalities")



#fars_data.list_tables()

#fars_data.table_schema("accident_2015")

fars_data.head("accident_2015")

#state_name

#minute_of_ems_arrival_at_hospital

#number_of_drunk_drivers

query = """SELECT *

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

         """



# check how big this query will be

fars_data.estimate_query_size(query) # 23 M



query_job = client.query(query)

iterator = query_job.result(timeout=30)

rows = list(iterator)

# Transform the rows into a nice pandas dataframe

df_far = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

#df_far.to_csv("accident-2015.csv", index = False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

#df_far = pd.read_csv("../input/fars-data.csv")

#df_far.head()

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

cat_vars = ['state_name', 'land_use_name', 'ownership_name', 'trafficway_identifier', 'first_harmful_event_name']

def topnrecords(df, n=5, column='score'):

     df = df.sort_values(by=column)[-n:]

     return df.sort_values(by = column,ascending = False)





for i, catvar in enumerate(cat_vars):

    df_temp = df_far[[catvar]]

    df_temp = df_temp.groupby([catvar]).size().to_frame().reset_index()

    df_temp.columns = [catvar, 'Total']

    df_x = topnrecords(df_temp, n = 10, column = "Total")

    plt.figure(i)

    plt.figure(figsize=(20,10))

    ax = sns.barplot(x=catvar, y="Total", data=df_x);

    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right");

    ax.set_title(catvar);
def topnrecords(df, n=5, column='score'):

     df = df.sort_values(by=column)[-n:]

     return df.sort_values(by = column,ascending = False)



df_all_states = df_far.copy()

df_all_states['Occurence'] = 1

df_high_states = df_all_states['Occurence'].groupby([df_all_states['state_name']]).count().reset_index()

df_x_15 = topnrecords(df_high_states, n = 15, column = "Occurence")

df_all_top_states = df_all_states[df_all_states.state_name.isin(df_x_15.state_name)]

df_all_states_grouped_years_type = df_all_top_states['Occurence'].groupby([df_all_top_states['state_name'], df_all_top_states['month_of_crash']]).count()

df_all_states_month_wise_accidents = df_all_states_grouped_years_type.reset_index()

plt.figure(figsize=(11,10))

sns.pointplot(x="month_of_crash", y="Occurence", hue="state_name", data=df_all_states_month_wise_accidents);
# Filtering on Texas and Florida

filter_states = ['Florida', 'Texas','California']

df_far_TX_FL_CA = df_far[df_far.state_name.isin(filter_states)]

df_3_states = df_far_TX_FL_CA.copy()

df_3_states['Occurence'] = 1

df_3_states_grouped_hours_type = df_3_states['Occurence'].groupby([df_3_states['state_name'], df_3_states['hour_of_crash']]).count()

df_3_states_hour_wise_accidents = df_3_states_grouped_hours_type.reset_index()

plt.figure(figsize=(11,8))

sns.barplot(x="hour_of_crash", y="Occurence", hue="state_name", data=df_3_states_hour_wise_accidents);
import warnings

warnings.filterwarnings('ignore')

g = sns.FacetGrid(df_3_states_hour_wise_accidents, col="state_name", size = 4)

g.map(sns.barplot,'hour_of_crash','Occurence');
df_all_states = df_far.copy()

df_all_states['Occurence'] = 1

df_all_states_grouped_hours_type = df_all_states['Occurence'].groupby([ df_all_states['hour_of_crash']]).count()

df_all_states_hour_wise_accidents = df_all_states_grouped_hours_type.reset_index()

df_all_states_hour_wise_accidents

plt.figure(figsize=(10,6))

sns.barplot(x="hour_of_crash", y = "Occurence" , data=df_all_states_hour_wise_accidents, color = "lightblue").set_title("All States");
import numpy as np

df_temp = df_far.copy()

df_temp = df_temp[df_temp.hour_of_crash.isin([0,1,2])]

df_far_TX_FL_3_hours = df_temp[['number_of_drunk_drivers', 'state_name', 'hour_of_crash']].copy()

grouped = df_far_TX_FL_3_hours.groupby(['state_name', 'hour_of_crash'])

df_result = grouped['number_of_drunk_drivers'].agg([np.sum,  np.size]).reset_index()

df_result = df_result.rename(columns={'size': 'count'})

df_result['drunk_driver_ratio'] = df_result['sum'] / df_result['count']
def topnrecords(df, n=5, column='score'):

     df = df.sort_values(by=column)[-n:]

     return df.sort_values(by = column,ascending = False)



#for hour_crash in df_result.hour_of_crash.unique():

for i, hour_crash in enumerate(df_result.hour_of_crash.unique()):

        #hour_crash = 2

        colname = "drunk_driver_ratio"

        df_temp = df_result[df_result.hour_of_crash == hour_crash]

        df_x = topnrecords(df_temp, n = 15, column = colname)

        df_x

        plt.figure(i)

        plt.figure(figsize=(20,10))

        ax = sns.barplot(x="state_name", y=colname, data=df_x);

        title = 'Drunk Drivers to accident ratio for hour: {}'.format(hour_crash);

        ax.set_title( title);
df_result = df_result[(df_result['count'] > 20)]

for i, hour_crash in enumerate(df_result.hour_of_crash.unique()):

        #hour_crash = 2

        colname = "drunk_driver_ratio"

        df_temp = df_result[df_result.hour_of_crash == hour_crash]

        df_x = topnrecords(df_temp, n = 15, column = colname)

        df_x

        plt.figure(i)

        plt.figure(figsize=(20,10))

        ax = sns.barplot(x="state_name", y=colname, data=df_x);

        title = 'Drunk Drivers to accident ratio for hour: {}'.format(hour_crash);

        ax.set_title( title);
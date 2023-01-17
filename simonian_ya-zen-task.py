# Add some libraries and import data



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# I imported data files to kaggle '/input' directory to have access from the kaggle notebook.

# Lets print all file data names



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Import all data sources



logs = pd.read_csv('/kaggle/input/zen_log_events.tsv', sep='\t')

events = pd.read_csv('/kaggle/input/app_metrika_events.tsv', sep='\t')

activations = pd.read_csv('/kaggle/input/activations.tsv', sep='\t')
# Visually inspect some elements from activations data



activations.head()
# Check uniqueness of the device_id for each data source



print('Activations rows {:,}, unique device_ids {:,}'.format(activations.shape[0], activations.nunique()['device_id']))

print('App metrika events rows {:,}, unique device_ids {:,}'.format(events.shape[0], events.nunique()['device_id']))

print('Logs rows {:,}, unique device_ids {:,}'.format(logs.shape[0], logs.nunique()['device_id']))



# We can see that device_id is a primary key for the activations table
# Join activations data with the app metrika data and

# visually inspect events list for device_id=696332390497458505 (choosen randomly)



activations_join = activations.join(events.set_index('device_id'), on='device_id', lsuffix='_activations', rsuffix='_events')

activations_join[activations_join["device_id"] == 696332390497458505].sort_values(by=['eventdate'])
# Visually inspect logs for one particular device_id=696332390497458505



activations_logs = activations.join(logs.set_index('device_id'), on='device_id', lsuffix='_activations', rsuffix='_logs')



# Inspect logs for all devices

# activations_logs[(activations_logs["device_id"] == 696332390497458505)].sort_values(by=['eventdate'])



# Inspect logs only for Zen App so that we can compare Log and App metrika events

activations_logs[(activations_logs["device_id"] == 696332390497458505) & (activations_logs["app"] == 'Zen App')].sort_values(by=['eventdate'])
# As we are going to measure channels effectiveness

# lets check how many unique channels we have



channels = activations['distr_channel'].unique().tolist()

print('Unique channels {}'.format(channels))
# For each device and date calculate sum of occured events



events1 = events[['device_id', 'eventdate', 'events_count']].groupby(by=['device_id', 'eventdate']).sum().reset_index()

events1.head()
# Build a pivot table to view device_id and dates on the different axises



active_devices = events[['device_id', 'eventdate', 'events_count']].pivot_table(index='device_id', columns='eventdate', values='events_count', aggfunc=np.sum, fill_value=0)



# We do not need events number for each day. We only whant to know if any

# event occured durint this date. So map table values to [0,1] values



active_devices[active_devices > 0] = 1

active_devices.head(10)
# Join events data to activations data to have full picture

# for each device activity for all days



active_devices = activations[['device_id', 'distr_channel', 'activation_date']].join(active_devices, on='device_id', lsuffix='_activations', rsuffix='_events').dropna()

active_devices.head(10)
# Calculate total number of active devices for each day (DAU)



channel_dau = active_devices.drop(columns=['device_id']).pivot_table(index='distr_channel', aggfunc=np.sum, fill_value=0)

channel_dau
# Plot DAU data for each channel



channel_dau.transpose().plot(title='DAU for Zen App', figsize=(12,8));
# We have to iterate all rows in events table so we define some helper functions



# This function takes row with time series and builds row with days relative to the activation_date 



def format_row(row):

    new_row = row.loc[['device_id', 'distr_channel', 'activation_date']]

    activation_date = row['activation_date']

    

    row.drop(labels=['device_id', 'distr_channel', 'activation_date'], inplace=True)

    

    for index, value in row.items():

        if pd.to_datetime(index) >= pd.to_datetime(activation_date):

            delta_days = (pd.to_datetime(index) - pd.to_datetime(activation_date)).days

            new_row.at[delta_days]=value

    

    return new_row



# This function iterates over the events table and calls format_row() for each row



def build_retention(input_df, limit=None):

    retention_df = pd.DataFrame()

        

    if(limit is None):

        # If limit is not set lets choose all rows

        limit = input_df.shape[0]

    else:

        # If limit is set we randomly choose rows

        input_df = input_df.sample(limit)



    i = 0

    for index, row in input_df.iterrows():

        i += 1

        new_row = format_row(row)

        retention_df = retention_df.append(new_row.to_frame().transpose())

        print('Progress {}% ({} from {}), Row length {}'.format(round(i*100/limit), i, limit, new_row.shape[0]))

        

    return retention_df.fillna(0)
print(active_devices.shape)



# Building retention table is time consuming so we are going to build it only

# for the subset of 1000 rows (full table will take 5 hours to build)

# We also output the progress and calculate the spent time



time_start = pd.Timestamp.now()

devices_retention = build_retention(active_devices, 1000)

time_end = pd.Timestamp.now()

time_delta = time_end - time_start

print('Process took {}'.format(time_delta))
# Visually inspect the resulting table



devices_retention.head()
# Some all activity values for each day to have aggregate value for each channel

# We take only 30 days as there are no activity values for later days



devices_retention1 = devices_retention.drop(columns=['device_id', 'activation_date']).iloc[:, :32].pivot_table(index='distr_channel', aggfunc=np.sum, fill_value=0)
# Calculate percentage values for retention relatively to the 0 day



devices_retention2 = devices_retention1.div(devices_retention1[0], axis=0).multiply(100).round(2)

devices_retention2.head()
# Plot retention graph for each channel



devices_retention2.transpose().plot(title='Retention for each channel', figsize=(12,8));
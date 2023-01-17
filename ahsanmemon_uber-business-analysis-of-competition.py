# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Separating the files that can't be opened because of unknown errors

valid_filenames = list()
invalid_filenames = list()

for filename in os.listdir("../input"):
    try:
        pd.read_csv('../input/'+ filename)
        valid_filenames.append('../input/'+ filename)
    except:
        invalid_filenames.append('../input/'+ filename)
        
print('Valid filenames')
print(pd.Series(valid_filenames))

print('\nInvalid filenames')
print(pd.Series(invalid_filenames))
# Reading All Files
# Uber Data
df_uber_april = pd.read_csv('../input/uber-raw-data-apr14.csv')
df_uber_may = pd.read_csv('../input/uber-raw-data-may14.csv')
df_uber_june = pd.read_csv('../input/uber-raw-data-jun14.csv')
df_uber_july = pd.read_csv('../input/uber-raw-data-jul14.csv')
df_uber_aug = pd.read_csv('../input/uber-raw-data-aug14.csv')
df_uber_sept = pd.read_csv('../input/uber-raw-data-sep14.csv')
df_uber_janjune_2015 = pd.read_csv('../input/uber-raw-data-janjune-15.csv')

# Other Companies Data
df_Dial7 = pd.read_csv('../input/other-Dial7_B00887.csv')
df_Lyft = pd.read_csv('../input/other-Lyft_B02510.csv')
df_Skyline = pd.read_csv('../input/other-Skyline_B00111.csv')
df_FHV = pd.read_csv('../input/other-FHV-services_jan-aug-2015.csv')
df_Federal = pd.read_csv('../input/other-Federal_02216.csv')
df_American = pd.read_csv('../input/other-American_B01362.csv')

# Renaming Time Columns for symmetry
df_uber_janjune_2015.rename(columns={'Pickup_date': 'Date/Time'}, inplace=True)
df_Dial7.rename(columns={'Date': 'Date/Time'}, inplace=True)
df_Lyft.rename(columns={'time_of_trip': 'Date/Time'}, inplace=True)
df_Skyline.rename(columns={'Date': 'Date/Time'}, inplace=True)
df_American.rename(columns={'DATE': 'Date/Time'}, inplace=True)
df_Federal.rename(columns={'Date': 'Date/Time'}, inplace=True)

# Standardizing Date Format
#     Saving Processing time by 40x by predefining format of date. Else it takes hours just to process dates 
df_uber_april = pd.DataFrame(pd.to_datetime(df_uber_april['Date/Time'],format = '%m/%d/%Y %H:%M:%S').dt.date)
df_uber_may = pd.DataFrame(pd.to_datetime(df_uber_may['Date/Time'],format = '%m/%d/%Y %H:%M:%S').dt.date)
df_uber_june = pd.DataFrame(pd.to_datetime(df_uber_june['Date/Time'],format = '%m/%d/%Y %H:%M:%S').dt.date)
df_uber_july = pd.DataFrame(pd.to_datetime(df_uber_july['Date/Time'],format = '%m/%d/%Y %H:%M:%S').dt.date)
df_uber_aug = pd.DataFrame(pd.to_datetime(df_uber_aug['Date/Time'],format = '%m/%d/%Y %H:%M:%S').dt.date)
df_uber_sept = pd.DataFrame(pd.to_datetime(df_uber_sept['Date/Time'],format = '%m/%d/%Y %H:%M:%S').dt.date)
df_uber_janjune_2015 = pd.DataFrame(pd.to_datetime(df_uber_janjune_2015['Date/Time'],format = '%Y/%m/%d %H:%M:%S').dt.date)
df_Dial7 = pd.DataFrame(pd.to_datetime(df_Dial7['Date/Time'],format = '%Y.%m.%d').dt.date)
df_Lyft = pd.DataFrame(pd.to_datetime(df_Lyft['Date/Time'],format = '%m/%d/%Y %H:%M').dt.date)
df_Skyline = pd.DataFrame(pd.to_datetime(df_Skyline['Date/Time'],format = '%m/%d/%Y').dt.date)
df_Federal = pd.DataFrame(pd.to_datetime(df_Federal[(df_Federal.Status == 'Arrived') | (df_Federal.Status == 'Assigned')]['Date/Time'],format = '%m/%d/%Y').dt.date)
df_American = pd.DataFrame(pd.to_datetime(df_American['Date/Time'],format = '%m/%d/%Y').dt.date)

# Adding Company Names for identification in the mixed dataset
df_uber_april['company'] = 'Uber'
df_uber_may['company'] = 'Uber'
df_uber_june['company'] = 'Uber'
df_uber_july['company'] = 'Uber'
df_uber_aug['company'] = 'Uber'
df_uber_sept['company'] = 'Uber'
df_uber_janjune_2015['company'] = 'Uber'
df_Dial7['company'] = 'Dial7'
df_Lyft['company'] = 'Lyft'
df_Skyline['company'] = 'Skyline'
df_Federal['company'] = 'Federal'
df_American['company'] = 'American'

# Combining All Data
df_all = pd.DataFrame()
# Uber Data
df_all = df_all.append(df_uber_april)
df_all = df_all.append(df_uber_may)
df_all = df_all.append(df_uber_june)
df_all = df_all.append(df_uber_july)
df_all = df_all.append(df_uber_aug)
df_all = df_all.append(df_uber_sept)
df_all = df_all.append(df_uber_janjune_2015)
# Other Services Data
df_all = df_all.append(df_Dial7)
df_all = df_all.append(df_Lyft)
df_all = df_all.append(df_Skyline)
df_all = df_all.append(df_Federal)
df_all = df_all.append(df_American)

# Sorting All Values
df_all.sort_values('Date/Time',inplace=True)

# Converting All Values to Pandas datetime
df_all['Date/Time'] = pd.to_datetime(df_all['Date/Time'])

# Adding column of month for visualizations
df_all['month'] = df_all['Date/Time'].dt.month
# Uber Business vs other businesses
start_date = '2014/01/01'
end_date = '2014/12/31'
df = df_all[(df_all['Date/Time']>=start_date) & (df_all['Date/Time']<=end_date)]
df.groupby(['month','company']).count().unstack('company')['Date/Time'].plot(kind='bar', figsize = (8,6),stacked=True)
plt.ylabel('Total Journeys')
plt.title('Uber Business vs Other Businesses (2014)');
# Individual Growth of Company Businesses in 2014
df.groupby(['month','company']).count().unstack('company')['Date/Time'].plot(figsize = (8,6),stacked=True)
plt.ylabel('Total Journeys')
plt.title('Growth of Company Businesses (2014)');
plt.grid()

# Uber did not seem to hurt the business of other companies in 2014
# as all other companies experienced a growth in their business along with Uber
# Uber Business in 2015
start_date = '2015/01/01'
end_date = '2015/12/31'
df = df_all[(df_all['Date/Time']>=start_date) & (df_all['Date/Time']<=end_date)]
df.groupby(['month','company']).count().unstack('company')['Date/Time'].plot(kind='bar', figsize = (8,6),stacked=True)
plt.ylabel('Total Journeys')
plt.title('Uber Rides (2015)');

# Uber rides continued to grow in the first half of 2015 as well
# Individual Growth of Company Business in 2015
df.groupby(['month','company']).count().unstack('company')['Date/Time'].plot(figsize = (8,6),stacked=True)
plt.ylabel('Total Journeys')
plt.title('Growth of Company Business (2015)');
plt.grid()
# P.S.
    # I have not included Files that were unreadable
    # I have also not included the file other-Federal_02216.csv because it didn't have individual ride data
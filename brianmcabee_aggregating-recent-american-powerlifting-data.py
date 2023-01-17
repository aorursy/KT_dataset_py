# Import Pandas

import pandas as pd



# Read csv file into a DataFrame

pl_df = pd.read_csv('../input/openpowerlifting-data-10162020/openpowerlifting-2020-10-16.csv',low_memory=True)
# Uncomment and run each line separately to view more information about entire dataframe

#pl_df.describe() # Show summary statistics of the data

#pl_df.head(5) # Show first 5 rows of dataset

#pl_df.shape

#pl_df.info()
# Make a new dataframe for raw, full power competitors in the USA only from year 2015 to 2020

sbd_df = pl_df.loc[(pl_df['Event'] == 'SBD') & 

                   (pl_df['Equipment'] == 'Raw') & 

                   (pl_df['MeetCountry'] == 'USA')]



# Drop unneeded columns

sbd_df = sbd_df.drop(['Country','Glossbrenner','Goodlift','MeetCountry'],axis=1)



# Convert Date column from string to datetime type

sbd_df['Date'] = pd.to_datetime(sbd_df['Date'])



# Remove data that is before 2015

sbd_df = sbd_df.loc[sbd_df['Date'].dt.year >= 2015]



# Drop duplicate values

sbd_df.drop_duplicates(inplace=True)



# run to verify duplicates are no longer there

# sum(sbd_df.duplicated())



# Sort Data by Date

sbd_df = sbd_df.sort_values('Date')



print('Full dataframe size: ' + str(int((pl_df.memory_usage().sum() / 1000000))) + 'MB')

print('American SBD dataframe size: ' + str(int((sbd_df.memory_usage().sum() / 1000000))) + 'MB')
#save sbd_df to csv

sbd_df.to_csv('usa_sbd_data_2020-10-16.csv')
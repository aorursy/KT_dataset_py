import numpy as np

import pandas as pd

from datetime import datetime
def func_process_datasets(path, to_del, to_datetime, to_int, to_cat, lat_lon):

    

    try:



        # Store strating time

        process_start = datetime.now()



        print('Processing {} ...\n'.format(path))



        # Load dataset into chuncks --------------------------------------------------------------------------------------------------------------------

        chunck_reader = pd.read_csv(path, chunksize = 500000)



        df_list = []

        i = 0



        for df in chunck_reader:

            df_list.append(df) # Append every chunk into df_list 

            i = i + 1

            print('Chunk: {} - length = {} - is processed and appended'.format(i, len(df)))



        print('\nAll chunks were processed and appended')

        print('All datasets loaded successfully\n\n')



        # Concat nested lists --------------------------------------------------------------------------------------------------------------------------

        main_df = pd.concat(df_list, sort = False)

        main_df.reset_index(drop = True, inplace = True)

        print('Main dataframe is created by concatenating all chunks')



        # Delete unwanted object to better memory usage

        del chunck_reader, df, df_list

        print('Release objects for better memory usage\n\n')



        # Store the length of the dataframe

        orig_df = len(main_df)  # original dataset length

        print('Dataset length is: {}'.format(orig_df))



        # Memory usage before optimization

        bo = round(main_df.memory_usage(deep = True).sum() / 1000**3, 2)   # Convert bytes to GBs

        print('DataFrame memory usage before optimization: {} GB\n\n'.format(bo))



        # Check numeric features missing values ---------------------------------------------------------------------------------------------------------

        main_df.loc[main_df.gender == -25, 'gender'] = np.nan

        main_df.loc[main_df.dpcapacity_start == -25, 'dpcapacity_start'] = np.nan

        main_df.loc[main_df.dpcapacity_end == -25, 'dpcapacity_end'] = np.nan

        main_df.loc[main_df.windchill == -999.0, 'windchill'] = np.nan

        main_df.loc[main_df.precipitation == -9999.0, 'precipitation'] = np.nan

        main_df.loc[main_df.wind_speed == -9999.0, 'wind_speed'] = np.nan

        main_df.loc[main_df.visibility == -9999.0, 'visibility'] = np.nan

        main_df.loc[main_df.pressure == -9999.0, 'pressure'] = np.nan

        main_df.loc[main_df.temperature == -9999.000000, 'temperature'] = np.nan

        main_df.loc[main_df.dewpoint == -9999.0, 'dewpoint'] = np.nan



        # Check categorical features missing values

        main_df.loc[main_df.events == 'unknown', 'events'] = np.nan

        main_df.loc[main_df.conditions == 'unknown', 'conditions'] = np.nan



        # Create list of the columns to test ------------------------------------------------------------------------------------------------------------

        numeric_cols = ['gender', 'dpcapacity_end', 'windchill', 'wind_speed', 'precipitation', 'visibility', 'pressure', 'temperature', 'dewpoint', 'events', 'conditions']



        for col in numeric_cols:

            missing_val = main_df[col].isnull().sum()

            print('{} feature contains: {} missing values. {}% of the dataset'.format(col, missing_val, round((missing_val / orig_df) * 100, 2)))



        # Drop unwanted columns -------------------------------------------------------------------------------------------------------------------------

        main_df.drop(to_del, axis = 1, inplace = True)

        print('\n{} features were droppped successfully\n\n'.format(to_del))



        # latitude_start: Assigning the 'Clark St & 9th St (AMLI)' LAT & LON for the missing values in latitude_start & longitude_start features ------------------------

        main_df.loc[main_df.latitude_start.isnull(), 'latitude_start'] = 41.870815

        main_df.loc[main_df.longitude_start.isnull(), 'longitude_start'] = -87.631248

        print('Missing start LAT & LON fixed')



        # latitude_end: Assigning the 'Clark St & 9th St (AMLI)' LAT & LON for the missing values in latitude_start & longitude_start features ------------------------

        main_df.loc[main_df.latitude_end.isnull(), 'latitude_end'] = 41.870815

        main_df.loc[main_df.longitude_end.isnull(), 'longitude_end'] = -87.631248

        print('Missing end LAT & LON fixed\n\n')



        # Fill nan values for specific features using the method = 'ffill'

        #df.loc[:,['temperature', 'pressure', 'pressure']].fillna(method= 'ffill', inplace=True)

        #print('['temperature', 'pressure', 'pressure'] features were treated\n\n')



        # Drop rows where they contain Nan values and reset the indexes

        main_df.dropna(inplace = True)

        main_df.reset_index(drop = True, inplace = True)

        print('Null values were dropped and indexes were reset successfully\n\n')



        # Encode classes ------------------------------------------------------------------------------------------------------------------------------------------------

        main_df.gender.replace({'Male': 0, 'Female': '1'}, inplace= True) 

        print('Gender feature encoded')



        # Process usertype encode classes

        main_df.usertype.replace({'Subscriber': 0, 'Customer': 1, 'Dependent': 3}, inplace= True)

        print('UserType feature encoded\n\n')



        # Converting datatypes ------------------------------------------------------------------------------------------------------------------------------------------

        print('Processing Datatype conversion ...\n')



        # Process int8 columns

        for col in to_int:

            main_df[col] = main_df[col].astype('int8')

            print('{} column processed...'.format(col))

        print('Int columns processed successfully\n')



        # Process category columns

        for col in to_cat:

            main_df[col] = main_df[col].astype('category')

            print('{} column processed...'.format(col))

        print('Category columns processed successfully\n')



        # Print note

        print('Processing Datetime features ...\n')    



        # Process datetime columns

        for col in to_datetime:

            main_df[col] = pd.to_datetime(main_df[col])

            print('{} column processed...'.format(col))

        print('Datetime columns processed successfully\n\n')



        # Process LAT & LON columns and convert them to float32

        for col in to_round4_latlon:

            main_df[col] = main_df[col].apply(lambda x: round(x,3))

            print('{} column processed...'.format(col))

        print('4 decimal points LAT & LON processed successfully\n')



        # Create new trip duration feature ------------------------------------------------------------------------------------------------------------------------------

        main_df['new_tripduration'] = main_df.stoptime - main_df.starttime

        main_df.new_tripduration = main_df.new_tripduration.astype('timedelta64[s]') # To convert 00:05:00 datetime to seconds

        main_df.new_tripduration = main_df.new_tripduration.astype('int16') # int8 will convert numbers and give negatove values, instead will use int16

        main_df.drop('tripduration', axis = 1, inplace= True) # Drop column

        print('[new_tripduration] feature created based on (stoptime - starttime) and the [tripduration] dropped')



        # Remove records where trip duration is less than 300 seconds - (1,268,968 Rows)

        main_df = main_df[(main_df.new_tripduration >= 300) & (main_df.new_tripduration <= 3600)] # Keep the data where trip duration is between 5 to 60 mins only.

        main_df.reset_index(drop= True, inplace= True)

        print('Data with trip duration between 5 to 60 mins kept\n\n')



        # Remove 2013 and 2014 from the dataset 

        main_df = main_df.loc[main_df.starttime.dt.year > 2014]

        main_df.sort_values(by = 'starttime', inplace = True)

        main_df.reset_index(drop = True, inplace = True)

        print('2013 and 2014 records dropped')

        print('New dataset length is: {} records. {} records dropped\n\n'.format(len(main_df), orig_df - len(main_df)))



        # Percentage of removed values ----------------------------------------------------------------------------------------------------------------------------------

        print('Dropped values percentage: {}%'.format(round(((orig_df - len(main_df)) / orig_df) * 100, 1)))



        # Memory usage before optimization ------------------------------------------------------------------------------------------------------------------------------

        ao = round(main_df.memory_usage(deep = True).sum() / 1000**3, 2)   # Convert bytes to GBs

        print('DataFrame memory usage after optimization: {} GB\n\n'.format(ao))



        # Store ending time

        process_end = datetime.now()



        # Finish statement

        print('Processing time: {}'.format(process_end - process_start))



        return main_df



    except:

        print('Error')
### Prepare lists

to_delete_cols   = ['trip_id', 'windchill', 'precipitation', 'dewpoint', 'from_station_id', 'to_station_id']

to_datetime_cols = ['starttime','stoptime']

to_category_cols = ['from_station_name','to_station_name','events','conditions']

to_round4_latlon = ['latitude_start', 'longitude_start', 'latitude_end', 'longitude_end']

to_int8_cols     = ['usertype', 'gender', 'dpcapacity_start', 'dpcapacity_end', 'tripduration', 'humidity', 'rain', 'dpcapacity_start', 'dpcapacity_end', 'temperature', 'pressure', 'visibility', 'wind_speed']
df = func_process_datasets( path = '../input/chicago-divvy-bicycle-sharing-data/data_raw.csv', 

                            to_del = to_delete_cols, 

                            to_datetime = to_datetime_cols,

                            to_int = to_int8_cols,

                            to_cat = to_category_cols,

                            lat_lon = to_round4_latlon )
df.to_pickle('divvy_bikeshare_picklefile.pickle')
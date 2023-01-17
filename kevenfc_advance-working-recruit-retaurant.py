import pandas as pd

import numpy as np
def describe_data(data):

    print("----------------------------------------------------------------")

    print("----- DATA REVIEW:")

    print("Length: {} rows, {} columns".format(len(data), len(data.columns)))

    print("\n----- DATA INFO:")

    print(data.info())

    print("\n----- MEMORY:")

    memory = data.memory_usage(index=True).sum()

    print("Memory consumed by training set : {} MB - {} KB" .format(memory/1024**2, memory/1024))

    '''

    print("\n----- VARIABLES:")

    for var in data.columns:

        print("--- VAR '{}':".format(var))

        number_values = len(data[var].unique())

        print("# Unique: {}".format(number_values))

        if(number_values < 20):

            print(data[var].unique())

        print("")

    '''
air_reserve = pd.read_csv("../input/air_reserve.csv")

air_reserve.head()
describe_data(air_reserve)
air_store_info = pd.read_csv("../input/air_store_info.csv")

air_store_info.head()
describe_data(air_store_info)
air_visit_data = pd.read_csv("../input/air_visit_data.csv")

air_visit_data.head()
describe_data(air_visit_data)
date_info = pd.read_csv("../input/date_info.csv")

date_info.head()
describe_data(date_info)
hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")

hpg_reserve.head()
describe_data(hpg_reserve)
hpg_store_info = pd.read_csv("../input/hpg_store_info.csv")

hpg_store_info.head()
describe_data(hpg_store_info)
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission.head()
sample_submission['air_store_id'] = sample_submission['id'].apply(lambda x: '_'.join(x.split('_')[:2]))

sample_submission['datetime'] = sample_submission['id'].apply(lambda x: '_'.join(x.split('_')[2:]))

sample_submission.head()

describe_data(sample_submission)
store_id_relation = pd.read_csv("../input/store_id_relation.csv")

store_id_relation.head()
describe_data(store_id_relation)
# Data: air_visit_data

air_visit_data['visit_date'] = pd.DatetimeIndex(air_visit_data['visit_date'])

# Data: air_reserve

air_reserve['visit_datetime'] = pd.DatetimeIndex(air_reserve['visit_datetime'])

air_reserve['visit_datetime'] = pd.DatetimeIndex(air_reserve['visit_datetime'])

# Data: hpg_reserve

hpg_reserve['visit_datetime'] = pd.DatetimeIndex(hpg_reserve['visit_datetime'])

hpg_reserve['visit_datetime'] = pd.DatetimeIndex(hpg_reserve['visit_datetime'])

# Data: air_visit_data

sample_submission['datetime'] = pd.DatetimeIndex(sample_submission['datetime'])

# Data: date_info

date_info['calendar_date'] = pd.DatetimeIndex(date_info['calendar_date'])
print("# unique values 'air_area_name':", len(air_store_info['air_area_name'].unique()))

print("# unique values 'hpg_area_name':",len(hpg_store_info['hpg_area_name'].unique()))
pd.merge(pd.concat([pd.DataFrame(air_store_info['air_area_name'].value_counts().index, columns = ['air_area_name']),

                    pd.DataFrame(air_store_info['air_area_name'].value_counts().values, columns = ['Restaurants'])], axis = 1), 

         pd.concat([pd.DataFrame(hpg_store_info['hpg_area_name'].value_counts().index, columns = ['hpg_area_name']),

                    pd.DataFrame(hpg_store_info['hpg_area_name'].value_counts().values, columns = ['Restaurants'])], axis = 1), 

         how="inner", 

         left_on = "air_area_name", right_on = "hpg_area_name")
print("# unique values 'air_genre_name':", len(air_store_info['air_genre_name'].unique()))

print("# unique values 'hpg_genre_name':",len(hpg_store_info['hpg_genre_name'].unique()))
pd.merge(pd.concat([pd.DataFrame(air_store_info['air_genre_name'].value_counts().index, columns = ['air_genre_name']),

                    pd.DataFrame(air_store_info['air_genre_name'].value_counts().values, columns = ['Restaurants'])], axis = 1), 

         pd.concat([pd.DataFrame(hpg_store_info['hpg_genre_name'].value_counts().index, columns = ['hpg_genre_name']),

                    pd.DataFrame(hpg_store_info['hpg_genre_name'].value_counts().values, columns = ['Restaurants'])], axis = 1), 

         how="inner", 

         left_on = "air_genre_name", right_on = "hpg_genre_name")
temp_air = air_store_info[['air_area_name','latitude','longitude']].drop_duplicates().reset_index(drop = True)

temp_air['lat_lon'] = air_store_info.apply(lambda x: '_'.join([str(x['latitude']),str(x['latitude'])]),axis = 1)

temp_air = pd.DataFrame(temp_air.groupby(['air_area_name'])['lat_lon'].count()).sort_values(by=['lat_lon'], ascending=[False])
temp_air.head(10)
temp_hpg = hpg_store_info[['hpg_area_name','latitude','longitude']].drop_duplicates().reset_index(drop = True)

temp_hpg['lat_lon'] = hpg_store_info.apply(lambda x: '_'.join([str(x['latitude']),str(x['latitude'])]),axis = 1)

temp_hpg_count = pd.DataFrame(temp_hpg.groupby(['hpg_area_name'])['lat_lon'].count()).sort_values(by=['lat_lon'], ascending=[False])
temp_hpg_count.head(10)
# Split areas to subareas:

air_store_info["area_level1"] = air_store_info.apply(lambda x: x['air_area_name'].split()[0],axis = 1)

air_store_info["area_level2"] = air_store_info.apply(lambda x: x['air_area_name'].split()[1],axis = 1)

air_store_info["area_level3"] = air_store_info.apply(lambda x: x['air_area_name'].split()[2],axis = 1)



hpg_store_info["area_level1"] = hpg_store_info.apply(lambda x: x['hpg_area_name'].split()[0],axis = 1)

hpg_store_info["area_level2"] = hpg_store_info.apply(lambda x: x['hpg_area_name'].split()[1],axis = 1)

hpg_store_info["area_level3"] = hpg_store_info.apply(lambda x: x['hpg_area_name'].split()[2],axis = 1)
air_store_info.head()
hpg_store_info.head()
air_store_info.head()
# Group by subareas in air_store_info

mae_df_air_area_level1 = air_store_info.groupby(["area_level1"], as_index = False, axis = 0)["air_store_id"].count()

mae_df_air_area_level2 = air_store_info.groupby(["area_level1", "area_level2"], 

                        as_index = False, axis = 0)["air_store_id"].count()

mae_df_air_area_level3 = air_store_info.groupby(["area_level1", "area_level2", "area_level3"], 

                        as_index = False, axis = 0)["air_store_id"].count()
mae_df_air_area_level1.rename(index=str, columns={"air_store_id": "count_as_level1"}, inplace = True)

mae_df_air_area_level1["key"] = mae_df_air_area_level1["area_level1"]

mae_df_air_area_level1.head()
mae_df_air_area_level2.rename(index=str, columns={"air_store_id": "count_as_level2"}, inplace = True)

mae_df_air_area_level2["key"] = mae_df_air_area_level2.apply(lambda x: '_'.join([str(x['area_level1']),str(x['area_level2'])]),axis = 1)

mae_df_air_area_level2.head()
mae_df_air_area_level3.rename(index=str, columns={"air_store_id": "count_as_level3"}, inplace = True)

mae_df_air_area_level3["key"] = mae_df_air_area_level3.apply(lambda x: '_'.join([str(x['area_level1']),

                                                                         str(x['area_level2']),

                                                                         str(x['area_level3'])]),axis = 1)

mae_df_air_area_level3.head()
# Group by ["latitude", "longitude"] in air_store_info

mae_df_air_lat_lon_store = air_store_info.groupby(["latitude", "longitude"], as_index=False, axis=0)["air_store_id"].count()

mae_df_air_lat_lon_store.head()
mae_df_air_lat_lon_store.rename(index=str, columns={"air_store_id": "count_as_lat_lon"}, inplace = True)

mae_df_air_lat_lon_store.head()
mae_df_air_genre = air_store_info.groupby(["latitude","longitude","air_genre_name"],

                                          as_index = False, axis = 0)["air_store_id"].count()

mae_df_air_genre.head()
mae_df_air_genre.rename(index = str, columns = {"air_store_id": "count_as_genre"}, inplace = True)

mae_df_air_genre.head()
# Resume mae_tables:

'''

mae_df_air_area_level1

mae_df_air_area_level2

mae_df_air_area_level3

mae_df_air_lat_lon_store

mae_df_air_genre

'''
# Group by subareas in hpg_store_info

mae_df_hpg_area_level1 = hpg_store_info.groupby(["area_level1"], as_index = False, axis = 0)["hpg_store_id"].count()

mae_df_hpg_area_level2 = hpg_store_info.groupby(["area_level1", "area_level2"], 

                        as_index = False, axis = 0)["hpg_store_id"].count()

mae_df_hpg_area_level3 = hpg_store_info.groupby(["area_level1", "area_level2", "area_level3"], 

                        as_index = False, axis = 0)["hpg_store_id"].count()
mae_df_hpg_area_level1.rename(index=str, columns={"hpg_store_id": "count_hpg_level1"}, inplace = True)

mae_df_hpg_area_level1["key"] = mae_df_hpg_area_level1["area_level1"]

mae_df_hpg_area_level1.head()
mae_df_hpg_area_level2.rename(index=str, columns={"hpg_store_id": "count_hpg_level2"}, inplace = True)

mae_df_hpg_area_level2["key"] = mae_df_hpg_area_level2.apply(lambda x: '_'.join([str(x['area_level1']),str(x['area_level2'])]),axis = 1)

mae_df_hpg_area_level2.head()
mae_df_hpg_area_level3.rename(index=str, columns={"hpg_store_id": "count_hpg_level3"}, inplace = True)

mae_df_hpg_area_level3["key"] = mae_df_hpg_area_level3.apply(lambda x: '_'.join([str(x['area_level1']),

                                                                         str(x['area_level2']),

                                                                         str(x['area_level3'])]),axis = 1)

mae_df_hpg_area_level3.head()
# Group by ["latitude", "longitude"] in hpg_store_info

mae_df_hpg_lat_lon_store = hpg_store_info.groupby(["latitude", "longitude"], as_index=False, axis=0)["hpg_store_id"].count()

mae_df_hpg_lat_lon_store.head()
mae_df_hpg_lat_lon_store.rename(index=str, columns={"hpg_store_id": "count_hpg_lat_lon"}, inplace = True)

mae_df_hpg_lat_lon_store.head()
mae_df_hpg_genre = hpg_store_info.groupby(["latitude","longitude","hpg_genre_name"],

                                          as_index = False, axis = 0)["hpg_store_id"].count()

mae_df_hpg_genre.head()
mae_df_hpg_genre.rename(index = str, columns = {"hpg_store_id": "count_hpg_genre"}, inplace = True)

mae_df_hpg_genre.head()
# Resume mae_tables:

'''

mae_df_hpg_area_level1

mae_df_hpg_area_level2

mae_df_hpg_area_level3

mae_df_hpg_lat_lon_store

mae_df_hpg_genre

'''
date_info
colname = 'calendar_date'

date_info[colname+"_year"] = date_info[colname].dt.year

date_info[colname+"_month"] = date_info[colname].dt.month

date_info[colname+"_day"] = date_info[colname].dt.day 

date_info[colname+"_weekday"] = date_info[colname].dt.weekday + 1

date_info[colname+"_hour"] = date_info[colname].dt.hour

date_info[colname+"_daycount"] = date_info[colname].apply(lambda x: x.toordinal())

date_info.loc[date_info[colname+"_weekday"]>=6, colname + '_weekend'] = 1

date_info.loc[date_info[colname+"_weekday"]<=5, colname + '_weekend'] = 0
date_info.head().T
date_info['holiday_flg'].value_counts()
'''Tables to trait:



air_reserve

air_visit_data

hpg_reserve

sample_submission



store_id_relation'''
#It will continue
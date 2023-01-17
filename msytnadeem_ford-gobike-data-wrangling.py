!pip install swifter
# import shutil

# shutil.rmtree("/kaggle/working")
import requests, zipfile, io

import pandas as pd

import numpy as np

import glob

import os

import matplotlib.pyplot as plt

import seaborn as sns

import swifter

from IPython.display import display



%matplotlib inline



sns.set(style="whitegrid")

pd.set_option('float_format', '{:f}'.format)

files = [

        {"year": "2018",

        "months": ['01','02','03','04','05','06','07','08', '09', '10','11','12'],

        "template_url" : 'https://s3.amazonaws.com/baywheels-data/%s%s-fordgobike-tripdata.csv.zip',

        "filetype":'zip'},

        {"year": "2019",

        "months": ['01','02','03','04'],

        "template_url" : 'https://s3.amazonaws.com/baywheels-data/%s%s-fordgobike-tripdata.csv.zip',

        "filetype":'zip'},

        {"year": "2019",

        "months": ['05','06','07','08', '09', '10'],

        "template_url" : 'https://s3.amazonaws.com/baywheels-data/%s%s-baywheels-tripdata.csv.zip',

        "filetype":'zip'},

    {"year":"2017",

        "months": [''], 

        "template_url" : 'https://s3.amazonaws.com/baywheels-data/%s%s-fordgobike-tripdata.csv',

         "filetype":'csv'},

]



# looping on the files based on the dictionary, read the stream in the memory, and extract it into data folder.



for file in files:

    for month in file["months"]:

        url = file["template_url"] % (file["year"], month)

        response = requests.get(url)

        if file["filetype"] == 'zip':

            z = zipfile.ZipFile(io.BytesIO(response.content))

            z.extractall('source')

        else:

            with open(os.path.join('source', "%s%s-fordgobike-tripdata.csv" % (file["year"], month)), mode='wb') as file:

                file.write(response.content)

        
# data_files = []

# for (dirpath, dirnames, filenames) in os.walk('/kaggle/working/source'):

#     data_files.extend(filenames)

#     break

# for file in data_files:

#     file_structure = file.split('-')

#     old = os.path.join("/kaggle/working/source", file)

#     print(file_structure)

#     new = os.path.join("/kaggle/working/source", file_structure[0]+'_'+file_structure[2])

#     os.rename(old,new)
path = r'/kaggle/working/source' # use your path

all_files = glob.glob(path + "/*.csv")



li = []



for filename in all_files:

    if filename == '/kaggle/working/source/201907-fordgobike-tripdata.csv':

        df = pd.read_csv(filename, index_col=None, sep=";",low_memory=False)

    else:

        df = pd.read_csv(filename, index_col=None,low_memory=False)

    li.append(df)
li_columns = ['duration_sec',

           'start_time','start_station_id','start_station_name', 'start_station_latitude','start_station_longitude',

           'end_time'  ,'end_station_id'  ,'end_station_name'  ,'end_station_latitude'   ,'end_station_longitude'  ,

           'bike_id','user_type','member_birth_year','member_gender','bike_share_for_all_trip','rental_access_method'

          ]
for i in li:

    deltacolumns = list(set(li_columns) - set(i.columns.tolist()))

    for column in deltacolumns:

        i[column] = np.nan
df = pd.concat(li, axis=0, ignore_index=True, sort=True)
df = df[li_columns].sort_values(by=['start_time', 'end_time'])
df.reset_index(inplace=True)
df[li_columns].to_csv('all_tripdata.csv')
df = pd.read_csv('all_tripdata.csv',low_memory=False)[li_columns]
df['start_station_id'] = df.start_station_id.astype('Int64').astype(str)

df['end_station_id'] = df.end_station_id.astype('Int64').astype(str)

df.loc[df['start_station_id'] == 'nan',['start_station_id']] = np.nan

df.loc[df['end_station_id'] == 'nan',['end_station_id']]  = np.nan
df['start_time'] = pd.to_datetime(df['start_time'], format="%Y-%m-%d %H:%M:%S.%f")

df['end_time'] = pd.to_datetime(df['end_time'], format="%Y-%m-%d %H:%M:%S.%f")
df['bike_id'] = df.bike_id.astype(str)

df['member_birth_year'] = df.member_birth_year.astype('Int64').astype(str)
display(df.head(10))

display(df.info())

display((df.isna().mean() * df.shape[0]).astype(int))

display(df.describe())
display(df.query('start_station_latitude == 0 or start_station_longitude == 0').start_station_id.unique())

display(df.query('end_station_latitude == 0 or end_station_longitude == 0').end_station_id.unique())
start_stations_locations = df.query("start_station_id not in ('420', '449')")[['start_station_id', 'start_station_latitude',"start_station_longitude"]]

end_stations_locations = df.query("end_station_id not in ('420', '449')")[['end_station_id', 'end_station_latitude',"end_station_longitude"]]

start_stations_locations.rename(columns={"start_station_id": "station_id", "start_station_latitude": "station_latitude", "start_station_longitude":"station_longitude"},inplace=True)

end_stations_locations.rename(columns={"end_station_id": "station_id", "end_station_latitude": "station_latitude", "end_station_longitude":"station_longitude"},inplace=True)

stations_locations = pd.concat([start_stations_locations,end_stations_locations])

stations_locations_max = stations_locations.groupby('station_id').max()

stations_locations_min = stations_locations.groupby('station_id').min()

stations_locations_max_min = stations_locations_max.join(stations_locations_min, lsuffix='_max', rsuffix='_min')

stations_locations_max_min_wo_nan = stations_locations_max_min.query('station_id != "nan"')

stations_locations_max_min_wo_nan = stations_locations_max_min_wo_nan.round(4)

stations_locations_max_min_wo_nan['matched_latitude'] = stations_locations_max_min_wo_nan.station_latitude_max == stations_locations_max_min_wo_nan.station_latitude_min

stations_locations_max_min_wo_nan['matched_longitude'] = stations_locations_max_min_wo_nan.station_longitude_max == stations_locations_max_min_wo_nan.station_longitude_min

stations_locations_max_min_wo_nan_unmatched = stations_locations_max_min_wo_nan.query('matched_latitude == False or matched_longitude == False')

stations_locations_max_min_wo_nan_unmatched.plot.scatter('station_longitude_max','station_latitude_max',figsize=(10,5), c="blue")

stations_locations_max_min_wo_nan_unmatched.plot.scatter('station_longitude_min','station_latitude_min',figsize=(10,5), c="red")
stations_locations_max_min_wo_nan_unmatched.query('station_latitude_max > 44')
stations_locations_max_min_wo_nan_unmatched_filtered = stations_locations_max_min_wo_nan_unmatched.query('station_latitude_max < 44')

stations_locations_max_min_wo_nan_unmatched_filtered.plot.scatter('station_longitude_max','station_latitude_max',figsize=(10,5),c="blue")

stations_locations_max_min_wo_nan_unmatched_filtered.plot.scatter('station_longitude_min','station_latitude_min',figsize=(10,5),c="red")
actual_duration_second = df[['duration_sec', 'start_time', 'end_time']].copy()

actual_duration_second['actual_duration_sec'] = (actual_duration_second['end_time'] - actual_duration_second['start_time']).dt.seconds

actual_duration_second.query('duration_sec != actual_duration_sec').shape[0]
display(

    df.member_gender.unique(),

    df.bike_share_for_all_trip.unique(),

    df.rental_access_method.unique())

df[["member_birth_year"]].astype(float).boxplot(column="member_birth_year")
df[["member_birth_year"]].astype(float).query('member_birth_year > 1960').boxplot(column="member_birth_year")
display(df[["member_birth_year"]].astype(float).query('member_birth_year <= 1960 and member_birth_year > 1940').boxplot(column="member_birth_year"))
display(df[["member_birth_year"]].astype(float).query('member_birth_year <= 1940').boxplot(column="member_birth_year"))
df_clean = df.copy()
display(df_clean.query('start_station_id == "420"'),df_clean.query('end_station_id == "420"'))
df_clean = df_clean.query('start_station_id != "420" and end_station_id != "420"')
display(df_clean.query('start_station_id == "449"'),df_clean.query('end_station_id == "449"'))
df_clean = df_clean.query('start_station_id != "449" and end_station_id != "449"')
display(df_clean.query('start_station_id == "408"'),df_clean.query('end_station_id == "408"'))
df_clean.loc[df_clean["end_station_id"] == "408", 

             ["start_station_latitude","start_station_longitude",

              "end_station_latitude","end_station_longitude"]] = [37.718513,-122.388320,37.718513, -122.388320 ]
start_stations_locations = df_clean[['start_station_id', 'start_station_latitude',"start_station_longitude"]].copy()

end_stations_locations = df_clean[['end_station_id', 'end_station_latitude',"end_station_longitude"]].copy()

start_stations_locations.rename(columns={"start_station_id": "station_id", "start_station_latitude": "station_latitude", "start_station_longitude":"station_longitude"},inplace=True)

end_stations_locations.rename(columns={"end_station_id": "station_id", "end_station_latitude": "station_latitude", "end_station_longitude":"station_longitude"},inplace=True)

stations_locations = pd.concat([start_stations_locations,end_stations_locations])

stations_locations_median = stations_locations.groupby('station_id').median()
start_nan_stations = df_clean.query('start_station_id != start_station_id').reset_index()[['index', 'start_station_latitude', 'start_station_longitude']]

end_nan_stations = df_clean.query('end_station_id != end_station_id').reset_index()[['index', 'end_station_latitude', 'end_station_longitude']]
def start_distance_calculation(row):

    delta_lat = (stations_locations_median['station_latitude'] - row['start_station_latitude']) ** 2

    delta_long = (stations_locations_median['station_longitude'] - row['start_station_longitude']) ** 2

    distance = np.sqrt(delta_lat + delta_long)

    new_cols = pd.Series(data=[distance.min(), distance.idxmin()], index=['start_min_distance_value', 'start_min_distance_station']) 

    result = pd.concat([row, new_cols])

    return result





start_nan_stations_nearest = start_nan_stations.swifter.apply(start_distance_calculation, axis=1)

start_nan_stations_nearest["index"] = start_nan_stations_nearest["index"].astype(int)
def end_distance_calculation(row):

    delta_lat = (stations_locations_median['station_latitude'] - row['end_station_latitude']) ** 2

    delta_long = (stations_locations_median['station_longitude'] - row['end_station_longitude']) ** 2

    distance = np.sqrt(delta_lat + delta_long)

    new_cols = pd.Series(data=[distance.min(), distance.idxmin()], index=['end_min_distance_value', 'end_min_distance_station']) 

    result = pd.concat([row, new_cols])

    return result





end_nan_stations_nearest = end_nan_stations.swifter.apply(end_distance_calculation, axis=1)

end_nan_stations_nearest["index"] = end_nan_stations_nearest["index"].astype(int)
start_nan_stations_accepted = start_nan_stations_nearest.query('start_min_distance_value < 0.01').set_index("index")[["start_min_distance_value", "start_min_distance_station"]]

end_nan_stations_accepted = end_nan_stations_nearest.query('end_min_distance_value < 0.01').set_index("index")[["end_min_distance_value", "end_min_distance_station"]]
df_clean.index.name = "index"
df_clean = df_clean.join(start_nan_stations_accepted)

df_clean.loc[df_clean["start_min_distance_value"] == df_clean["start_min_distance_value"], ["start_station_id"]]= df_clean["start_min_distance_station"]

df_clean.drop(columns=['start_min_distance_value', 'start_min_distance_station'],inplace=True)
df_clean = df_clean.join(end_nan_stations_accepted)

df_clean.loc[df_clean["end_min_distance_value"] == df_clean["end_min_distance_value"], ["end_station_id"]]= df_clean["end_min_distance_station"]

df_clean.drop(columns=['end_min_distance_value', 'end_min_distance_station'],inplace=True)
df_clean = df_clean.query('start_station_id == start_station_id')

df_clean = df_clean.query('end_station_id == end_station_id')
def fix_station_lat_long(row):

    df_clean.loc[df_clean["start_station_id"] == row["station_id"], ["start_station_latitude","start_station_longitude"]]  =  [row.station_latitude,row.station_longitude]

    df_clean.loc[df_clean["end_station_id"]   == row["station_id"], ["end_station_latitude","end_station_longitude"]]    =  [row.station_latitude, row.station_longitude]



stations_locations_median_rounded_4 = stations_locations_median.reset_index().round(4)    

_ignore = stations_locations_median_rounded_4.swifter.apply(fix_station_lat_long, axis=1)
start_stations_locations_clean = df_clean[['start_station_id', 'start_station_latitude',"start_station_longitude"]].copy()

end_stations_locations_clean = df_clean[['end_station_id', 'end_station_latitude',"end_station_longitude"]].copy()

start_stations_locations_clean.rename(columns={"start_station_id": "station_id", "start_station_latitude": "station_latitude", "start_station_longitude":"station_longitude"},inplace=True)

end_stations_locations_clean.rename(columns={"end_station_id": "station_id", "end_station_latitude": "station_latitude", "end_station_longitude":"station_longitude"},inplace=True)

stations_locations_clean = pd.concat([start_stations_locations_clean,end_stations_locations_clean])

stations_locations_max_clean = stations_locations_clean.groupby('station_id').max()

stations_locations_min_clean = stations_locations_clean.groupby('station_id').min()

stations_locations_max_min_clean = stations_locations_max_clean.join(stations_locations_min_clean, lsuffix='_max', rsuffix='_min')

stations_locations_max_min_wo_nan_clean = stations_locations_max_min_clean.query('station_id != "nan"').round(4).copy()

stations_locations_max_min_wo_nan_clean['matched_latitude'] = stations_locations_max_min_wo_nan_clean.station_latitude_max == stations_locations_max_min_wo_nan_clean.station_latitude_min

stations_locations_max_min_wo_nan_clean['matched_longitude'] = stations_locations_max_min_wo_nan_clean.station_longitude_max == stations_locations_max_min_wo_nan_clean.station_longitude_min

stations_locations_max_min_wo_nan_unmatched_clean = stations_locations_max_min_wo_nan_clean.query('matched_latitude == False or matched_longitude == False')

stations_locations_max_min_wo_nan_unmatched_clean
main_map_edges = (-122.5,-121.8,37.2,37.9)

main_map_image = plt.imread('../input/map-images/map.png')

fig, ax = plt.subplots(figsize = (10,10))

ax.scatter(stations_locations_max_min_clean.station_longitude_max,stations_locations_max_min_clean.station_latitude_max,zorder=1, alpha= 0.8, c='b', s=20)

ax.set_xlim(main_map_edges[0],main_map_edges[1])

ax.set_ylim(main_map_edges[2],main_map_edges[3])

ax.imshow(main_map_image, zorder=0, extent = main_map_edges, aspect= 'equal')
df_clean["start_station_city"] = np.nan

df_clean["end_station_city"] = np.nan

df_clean.loc[df_clean['start_station_longitude'] < -122.35, "start_station_city"] = "SAN FRANSISCO"

df_clean.loc[(df_clean['start_station_longitude'] > -122.35) & (df_clean['start_station_longitude'] < -122.1 ), "start_station_city"] = "OAKLAND"

df_clean.loc[df_clean['start_station_longitude'] > -122.1 , "start_station_city"] = "SAN JOSE"

df_clean.loc[df_clean['end_station_longitude'] < -122.35, "end_station_city"] = "SAN FRANSISCO"

df_clean.loc[(df_clean['end_station_longitude'] > -122.35) & (df_clean['end_station_longitude'] < -122.1 ), "end_station_city"] = "OAKLAND"

df_clean.loc[df_clean['end_station_longitude'] > -122.1 , "end_station_city"] = "SAN JOSE"
df_clean.query('start_station_city != end_station_city').start_station_city.count()
start_stations_names = df_clean[['start_station_id', 'start_station_name']].copy()

end_stations_names = df_clean[['end_station_id', 'end_station_name']].copy()

start_stations_names.rename(columns={"start_station_id": "station_id", "start_station_name": "station_name"},inplace=True)

end_stations_names.rename(columns={"end_station_id": "station_id", "end_station_name": "station_name"},inplace=True)

stations_names = pd.concat([start_stations_names,end_stations_names])

stations_names_wo_duplicates = stations_names.drop_duplicates()

stations_many_names = stations_names_wo_duplicates.groupby('station_id').count().query('station_name > 1')
stations_many_names.shape[0]
stations_names_requiers_offline_fix = stations_names_wo_duplicates.query(f'station_id in {stations_many_names.index.tolist()}').sort_values('station_id').set_index('station_id').query('station_name == station_name')

stations_names_requiers_offline_fix.to_csv('stations_names_error.csv')
station_names_fixed = pd.read_csv("../input/visual-assessment/stations_names_fixed.csv")
station_names_fixed["station_id"] = station_names_fixed.station_id.astype(str)
stations_names_wo_duplicates_fixed = stations_names_wo_duplicates.query('station_name == station_name').set_index('station_id').join(station_names_fixed.set_index('station_id'),rsuffix=('_fixed')).reset_index()

stations_names_wo_duplicates_fixed.loc[stations_names_wo_duplicates_fixed['station_name_fixed'] == stations_names_wo_duplicates_fixed['station_name_fixed'], "station_name"] = stations_names_wo_duplicates_fixed['station_name_fixed'] 

stations_names_wo_duplicates_fixed = stations_names_wo_duplicates_fixed.drop_duplicates().set_index("station_id")[["station_name"]]
def fix_station_names(row):

    df_clean.loc[df_clean["start_station_id"] == row["station_id"], ["start_station_name"]]  =  [row.station_name]

    df_clean.loc[df_clean["end_station_id"]   == row["station_id"], ["end_station_name"]]    =  [row.station_name]



_ignore = stations_names_wo_duplicates_fixed.reset_index().swifter.apply(fix_station_names, axis=1)
df_clean.loc[:,['duration_sec']] = (df_clean['end_time'] - df_clean['start_time']).dt.seconds
actual_duration_second = df_clean[['duration_sec', 'start_time', 'end_time']].copy()

actual_duration_second['actual_duration_sec'] = (actual_duration_second['end_time'] - actual_duration_second['start_time']).dt.seconds

actual_duration_second.query('duration_sec != actual_duration_sec').shape[0]
df_clean.duration_sec.describe()
df_clean.member_gender.replace('M','Male', inplace=True)

df_clean.member_gender.replace('F','Female', inplace=True)

df_clean.member_gender.replace('O','Other', inplace=True)

df_clean.member_gender.replace('?','Other', inplace=True)

df_clean.member_gender.fillna("Other", inplace=True)
df_clean.member_gender.unique()
df_clean.member_birth_year = df_clean.member_birth_year.astype(float)

df_clean.boxplot(column="member_birth_year")
df_clean.query('member_birth_year < 1960').member_birth_year.hist(bins=30)
df_clean.query('member_birth_year < 1960 and member_birth_year > 1945').boxplot(column="member_birth_year")
df_clean.query('member_birth_year <= 1945').boxplot(column="member_birth_year")
df_clean.query('member_birth_year < 1945').member_birth_year.hist(bins=20)
df_clean.query('member_birth_year < 1945').shape[0]
df_clean.loc[df_clean["member_birth_year"] < 1945, "member_birth_year" ] = np.nan

df_clean.query('member_birth_year < 1945').shape[0]
df_clean.boxplot(column="member_birth_year")
df_clean["member_age_group"] = np.nan
df_clean.loc[(df_clean["member_birth_year"] <= 2001) & (df_clean["member_birth_year"] > 1992), "member_age_group"] = "18-26"

df_clean.loc[(df_clean["member_birth_year"] <= 1992) & (df_clean["member_birth_year"] > 1979), "member_age_group"] = "27-39"

df_clean.loc[(df_clean["member_birth_year"] <= 1979) & (df_clean["member_birth_year"] > 1962), "member_age_group"] = "40-57"

df_clean.loc[(df_clean["member_birth_year"] <= 1862), "member_age_group"] = "older than 57"
df_clean.head(10)
(df_clean.isna().mean() * df_clean.shape[0]).astype(int)
df_clean.to_csv('all_tripdata_cleaned.csv', index=False)
from subprocess import call

call(['python', '-m', 'nbconvert', 'Project5-Data Visualization _ Wrangling.ipynb'])
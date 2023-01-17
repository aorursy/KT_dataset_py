import gc

import os, sys

import random



import requests, json

import re



import numpy as np

import pandas as pd

import seaborn as sns



from matplotlib import pyplot as plt



path_data = "/kaggle/input/ashrae-energy-prediction/"

TRAIN_FILE = path_data + "train.csv"

path_test = path_data + "test.csv"

TRAIN_BUILDING_FILE = path_data + "building_metadata.csv"

path_weather_train = path_data + "weather_train.csv"

path_weather_test = path_data + "weather_test.csv"



TEST_SCRAP = "/kaggle/input/ucf-v0"
## Memory optimization

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def df_optimization(df, use_float16=False, verbose=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")

        

        if verbose:

            print("col: %s was %s and is %s" % (col, col_type, df[col].dtype))

    

    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
UCF_BUILDINGS = {77: [131, "Polk Hall"]}
# UCF data from https://www.oeis.ucf.edu/buildings

ucl_pd = pd.read_csv(TEST_SCRAP + "/site_0.csv", sep=";")  

ucl_pd["ucl_id"] = ucl_pd["URL"].apply(lambda x: x.rsplit('/', 1)[-1])

ucl_pd["ucl_id"] = ucl_pd["ucl_id"].astype(np.int)

ucl_pd["EUI"] = ucl_pd["EUI"].astype(np.float32)

ucl_pd.head()
train_building_pd = pd.read_csv(TRAIN_BUILDING_FILE) 

train_building_pd = df_optimization(train_building_pd)

train_building_pd.head()
train_pd = pd.read_csv(TRAIN_FILE)

train_pd["timestamp"] = pd.to_datetime(train_pd["timestamp"], format="%Y-%m-%d %H:%M:%S")

train_pd = df_optimization(train_pd)

train_pd.head()
def plot_heatmap(df, agg_col, agg_function, figsize=(12, 10)):

    df_pd = df.groupby(['site_id', 'primary_use']).agg({agg_col: [agg_function]})

    df_pd.columns = ['_'.join(col).strip() for col in df_pd.columns.values]

    name = agg_function if isinstance(agg_function, str) else agg_function.__name__

    df_pd = df_pd.reset_index().pivot(index='primary_use', columns='site_id', values='%s_%s' % (agg_col, name))

    fig, ax = plt.subplots(figsize=figsize)

    d = sns.heatmap(data=df_pd, annot=True, fmt='.1f', cmap="BuPu", ax=ax)

    d = plt.title('%s_%s' % (agg_col, name))

    plt.show()    
plot_heatmap(train_building_pd, "building_id", 'count')  
def scrap_ucf_building(building_id, ucf_building_id, save_file=None):

    final_building_pd = None

    payload = {'resolution': 'hour', 'building': '%s' % ucf_building_id, 'filetype': 'json', 'start-date': "01/01/2016", 'end-date': '01/01/2019'}

    r = requests.post("https://www.oeis.ucf.edu/getData", data=payload)

    if (r.status_code == 200) & (r.headers['Content-Type'] == 'application/json'):

        b = r.json()

        if save_file:

            with open(save_file, 'w') as outfile:

                json.dump(b, outfile)            

        for m in range(len(b)):

            meter = b[m]['key']

            print("Loading building %d/%d %s" % (ucf_building_id, building_id, meter))

            temp_pd = pd.DataFrame(b[m]['values'])

            temp_pd["meter"] = meter

            temp_pd["building_id"] = building_id

            temp_pd.rename(columns={"reading": "meter_reading"}, inplace=True)

            if final_building_pd is None:

                final_building_pd = temp_pd

            else:

                final_building_pd = pd.concat([final_building_pd, temp_pd], axis=0)

            

        final_building_pd["timestamp"] = pd.to_datetime(final_building_pd["timestamp"], format="%Y-%m-%d %H:%M:%S")

        return final_building_pd

    else:

        return None
def scrap_details(url, uid):

    r = requests.get(url)

    built_year = None

    long = None

    lat = None

    if r.text.find("constructed in") > 0:

        result = re.search('constructed in (.*)</div>', r.text)

        if result is not None:

            built_year = result.group(1)

        result = re.search("\"lng\": '(.*)',", r.text)

        if result is not None:

            long = result.group(1)

        result = re.search("\"lat\": '(.*)',", r.text)

        if result is not None:

            lat = result.group(1)

    return (uid, built_year, lat, long)
# Get year, lat, long of each building

results = []

for idx, row in ucl_pd.iterrows():

    d = scrap_details(row["URL"], row["ucl_id"])

    results.append(d)

year_pd = pd.DataFrame(results, columns = ["ucl_id", "year", "lat", "long"])

year_pd["year"] = year_pd["year"].astype(np.float32)

year_pd.head()
ucl_full_pd = pd.merge(ucl_pd, year_pd, on=["ucl_id"], how="left")

ucl_full_pd.head()
# Map correct primary_use

UCL_TYPE_MAP = {

    "Residence Hall": "Lodging/residential",

    "Classroom": "Education",

    "Stadium": "Entertainment/public assembly",

    "Parking Garage": "Parking",

    "Research": "Education"

}

ucl_full_pd["Type"] = ucl_full_pd["Type"].map(UCL_TYPE_MAP)
# Find candidates by joining on ["square_feet", "primary_use", "year_built"]

train_building_candidate_pd = pd.merge(train_building_pd.query("site_id == 0"), ucl_full_pd, left_on=["square_feet", "primary_use", "year_built"], right_on=["Area", "Type", "year"], how="left").dropna(subset =["Name"])

train_building_candidate_pd["candidates"] = train_building_candidate_pd.groupby("building_id", as_index=False)["ucl_id"].transform('count')

train_building_candidate_pd["ucl_id"] = train_building_candidate_pd["ucl_id"].astype(int)

train_building_candidate_pd.query("candidates == 1").head()
# Update UCF_BUILDINGS dictionary from candidates

for idx, row in train_building_candidate_pd.query("candidates == 1").iterrows():

    building_id = row["building_id"]

    ucl_id  = row["ucl_id"]

    name = row["Name"]

    UCF_BUILDINGS[building_id] = [ucl_id, name]
# UCF API calls to get data

final_building_scrap_pd = None

for key, value in UCF_BUILDINGS.items():

    building_scrap_pd = scrap_ucf_building(key, value[0], save_file=None)

    if final_building_scrap_pd is None:

        final_building_scrap_pd = building_scrap_pd

    else:

        final_building_scrap_pd = pd.concat([final_building_scrap_pd, building_scrap_pd], axis=0)
final_building_scrap_pd = final_building_scrap_pd.set_index('timestamp').sort_index()
# Meter found

final_building_scrap_pd["meter"].unique()
# We don't need Irrigation, Water, Gas

final_building_scrap_pd["meter"] = final_building_scrap_pd["meter"].map({'Electric':0, 'Chilled Water': 1, 'Irrigation':-1, 'Water': -2, 'Gas': -3})
train_pd.head()
# Plot Electricity

validated_ids = []

for b_id in final_building_scrap_pd["building_id"].unique():

    

    scrap = final_building_scrap_pd.query("meter == 0 & building_id == %d & timestamp < '2017-01-01 00:00:00'" % b_id).reset_index().set_index('timestamp').sort_index()["meter_reading"].values

    hist = train_pd.query("meter == 0 & building_id == %d" % b_id).set_index('timestamp').sort_index()["meter_reading"].values

    

    if len(scrap) != len(hist):

        print("\n**** Building id = %d, not same length scrap = %d train = %d ****" %(b_id, len(scrap), len(hist)))

        validated_ids.append((b_id, len(scrap), len(hist)))

    else:

        diff = np.nansum(hist - scrap)

        print("Building id = %d, Diff = %d" %(b_id, diff))

        validated_ids.append((b_id, len(scrap), len(hist)))



    fig, ax = plt.subplots(figsize=(20, 4))

    d = final_building_scrap_pd.query("meter == 0 & building_id == %d" % b_id).plot(kind='line', y=["meter_reading"], ax=ax, linestyle='-', linewidth=0.5)

    d = train_pd.query("meter == 0 & building_id == %d" % b_id).set_index('timestamp').plot(kind='line', y="meter_reading", ax=ax, alpha=0.5, linewidth=1.0, title="meter: %d building_id: %d" % (0, b_id))

    plt.show()
# To improve by fixing different length (see valid == 0)

validated_pd = pd.DataFrame(validated_ids, columns=["building_id", "scrap_len", "hist_len"])

validated_pd["valid"] = np.where(validated_pd["scrap_len"] == validated_pd["hist_len"], 1, 0)

validated_pd.query("valid == 1").head()
# Not perfect match

validated_pd.query("valid == 0").head()
# Chilled water match with a coefficient, let's compute it

CW_COEF = final_building_scrap_pd.query("meter == 1 & building_id == %d & timestamp > '2016-05-21'" % 60)["meter_reading"].values[0]/train_pd.query("meter == 1 & building_id == %d & timestamp > '2016-05-21'" % 60)["meter_reading"].values[0]

print(CW_COEF)
validated_cw_ids = []

for b_id in final_building_scrap_pd["building_id"].unique():

    if len(final_building_scrap_pd.query("meter == 1 & building_id == %d" % b_id)) > 0:

        

        scrap = final_building_scrap_pd.query("meter == 1 & building_id == %d & timestamp < '2017-01-01 00:00:00'" % b_id).dropna().reset_index().set_index('timestamp').sort_index()["meter_reading"].values

        hist = train_pd.query("meter == 1 & building_id == %d" % b_id).set_index('timestamp').sort_index()["meter_reading"].values



        if len(scrap) != len(hist):

            print("\n**** Building id = %d, not same length scrap = %d train = %d ****" %(b_id, len(scrap), len(hist)))

            validated_cw_ids.append((b_id, len(scrap), len(hist)))

        else:

            diff = np.nansum(hist - scrap/CW_COEF)

            print("Building id = %d, Diff = %d" %(b_id, diff))

            validated_cw_ids.append((b_id, len(scrap), len(hist)))

        

        fig, ax = plt.subplots(figsize=(20, 4))

        tmp_pd = final_building_scrap_pd.query("meter == 1 & building_id == %d" % b_id).copy()

        tmp_pd["meter_reading"] = tmp_pd["meter_reading"]/CW_COEF

        d = tmp_pd.plot(kind='line', y=["meter_reading"], ax=ax)

        d = train_pd.query("meter == 1 & building_id == %d" % b_id).set_index('timestamp').plot(kind='line', y=["meter_reading"], ax=d, alpha=0.5, title="meter: %d building_id: %d" % (1, b_id))

        plt.show()
# Keep validated data, clean/reindex/sort

completed_scrap_pd = final_building_scrap_pd.query("meter == 0 | meter == 1").reset_index().set_index(["building_id", "meter", "timestamp"]).sort_index().reset_index().copy()

completed_scrap_pd["meter_reading"] = np.where(completed_scrap_pd["meter"] == 1, completed_scrap_pd["meter_reading"]/CW_COEF, completed_scrap_pd["meter_reading"])

completed_scrap_pd.to_pickle("ucf_2016_2017_2018.pkl")
# Building that matches

matched = completed_scrap_pd["building_id"].unique()

matched
# The following have multiple matches so it should be done manually/

ucf_matches = set()

ucf_matches.add(131)

for m in matched:

    ucf_matches.add(UCF_BUILDINGS[m][0])

manual_pd = train_building_candidate_pd[(train_building_candidate_pd["candidates"] > 1) & (~train_building_candidate_pd["building_id"].isin(matched)) & (~train_building_candidate_pd["ucl_id"].isin(ucf_matches))]

manual_pd
manual_pd["building_id"].nunique()
train_building_candidate_pd.to_csv("building_candidate.csv")
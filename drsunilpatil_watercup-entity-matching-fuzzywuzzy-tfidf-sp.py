import geopandas as gpd

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_mh_all_villages = gpd.read_file('../input/mh-villages-v2w2/MH_Villages v2W2.shp')[['DTNAME','GPNAME','VILLNAME']]

# ['DTNAME','GPNAME','VILLNAME']

print(df_mh_all_villages.shape)

df_mh_all_villages.T
df_mh_all_villages["DTNAME"].unique()
print(len(df_mh_all_villages["DTNAME"].unique()))

df_mh_all_villages[df_mh_all_villages["DTNAME"]=="Sangali"]
df_mh_all_villages[df_mh_all_villages["DTNAME"]=="Mumbai"]
df_mh_all_villages[df_mh_all_villages["VILLNAME"].isnull()].shape
# We need to get rid of rows with missing village name
# Are the village names unique given a district?

df_mh_all_villages.groupby("DTNAME")["VILLNAME"].agg(["count","nunique"])
df_ListOfTalukas = pd.read_csv("../input/paani-foundations-satyamev-jayate-water-cup/ListOfTalukas.csv")

print(df_ListOfTalukas.shape)

df_ListOfTalukas.T
df_ListOfTalukas["District"].unique()
print("Number of unique districts",len(df_ListOfTalukas["District"].unique()))

df_ListOfTalukas[df_ListOfTalukas["District"]=="Sangli"]
df_StateLevelWinners = pd.read_csv('/kaggle/input/paani-foundations-satyamev-jayate-water-cup/StateLevelWinners.csv')

print(df_StateLevelWinners.shape)

df_StateLevelWinners.T
df_StateLevelWinners["District"].unique()
from fuzzywuzzy import fuzz



districts = df_mh_all_villages["DTNAME"].unique().tolist()



def get_best_district_match(mydist, districts = districts ):    

    fuzz_ratio = [fuzz.ratio(mydist, dist) for dist in districts]

    max_ratio = max(fuzz_ratio)

    idx_max = [i for i, j in enumerate(fuzz_ratio) if j == max_ratio]

    #ToDo: if more than one match throw an error

    return districts[idx_max[0]]    
get_best_district_match("Sangli")
df_StateLevelWinners["district_m"] = df_StateLevelWinners["District"].apply(lambda x:get_best_district_match(x))
_idx = df_StateLevelWinners["District"] != df_StateLevelWinners["district_m"]

df_StateLevelWinners.loc[_idx,["District","district_m"]]
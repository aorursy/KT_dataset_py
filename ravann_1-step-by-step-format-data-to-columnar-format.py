import os
idir = "../input/"
for x in os.listdir(idir):
        f = idir + x
        s = os.stat(f)
        num_lines = sum(1 for line in open(f))
        print(x + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")

rows = 0
for line in open(f): 
    print(line)
    rows += 1
    if rows > 4:
        break
import numpy as np 
import pandas as pd 
ifile = idir + "train.csv"
num_rows = 3
df = pd.read_csv(ifile, nrows = num_rows, converters={'fullVisitorId': str})
df.head()
df.info()
import json
j = json.loads(df["totals"][0])
j
from pandas.io.json import json_normalize
json_normalize(j)
json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource'] # List columns where data is stored in JSON format

# Apply converter to convert JSON format data into JSON object
json_conv = {col: json.loads for col in (json_cols)}

# Read the CSV with the new converter
df = pd.read_csv(idir + "train.csv", 
    dtype={'fullVisitorId': str},
    converters=json_conv, 
    nrows=num_rows)

df.head()
tdf = json_normalize(df["totals"])
tdf.head()
df = df.drop(columns = ["totals"])
df.head()
tdf.columns = ["totals_" + col for col in tdf.columns]
df = df.merge(tdf, left_index=True, right_index=True)
df.head()
def ld_dn_df(csv_file, json_cols, rows_to_load = 100 ): 

    # Apply converter to convert JSON format data into JSON object
    json_conv = {col: json.loads for col in (json_cols)}

    # Read the CSV with the new converter
    df = pd.read_csv(csv_file, 
        dtype={'fullVisitorId': str},
        converters=json_conv, 
        nrows=rows_to_load, 
        low_memory = False
        )
    
    for jcol in json_cols: 
        tdf = json_normalize(df[jcol])
        tdf.columns = [jcol + "_" + col for col in tdf.columns]
        df = df.merge(tdf, left_index=True, right_index=True)
        
    df = df.drop(columns = json_cols)
    return df
%%time
rows_to_load = 1000000
json_cols = ["totals", "device", "geoNetwork", "trafficSource"]
train_df =  ld_dn_df("../input/train.csv", json_cols, rows_to_load)
%%time
test_df =  ld_dn_df("../input/test.csv", json_cols, rows_to_load)
def replace_def_vals(df):
    df.replace({'(not set)': np.nan,
               'not available in demo dataset': np.nan,
               '(not provided)': np.nan,
               'unknown.unknown': np.nan,
               '(none)':np.nan,
               '/':np.nan,
               'Not Socially Engaged':np.nan},
              inplace=True)
%%time
replace_def_vals(train_df)
%%time
replace_def_vals(test_df)
f = './train_flat.csv'
train_df.to_csv(f, index = False)
s = os.stat(f)
num_lines = sum(1 for line in open(f))
print(x + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")
f = './test_flat.csv'
test_df.to_csv(f, index = False)
s = os.stat(f)
num_lines = sum(1 for line in open(f))
print(x + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")

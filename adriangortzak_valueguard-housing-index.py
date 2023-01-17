import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

import geopandas as gpd
# Get data 

df = pd.read_excel ('http://dataservice.valueguard.se/ExcelServlet/hoxIndex/index')



last_row_in_file = df.index[df['Month'].isnull()].tolist()[0]

df = df[0:last_row_in_file]

df['index_date'] = df['Month'].astype(str).str[:4] + "-" + df['Month'].astype(str).str[4:6]  + "-15"

df['index_date'] = pd.to_datetime(df['index_date'])

df.set_index('index_date', inplace=True)
df.iloc[-1]
df[['HOXSWE','HOXHOUSESWE','HOXFLATSWE','HOXHOUSEMCSWE']].plot.line(figsize=(30,10))
df[['HOXFLATSTO','HOXHOUSESTO']].plot.line(figsize=(30,10))
df[['HOXFLATGBG','HOXHOUSEGBG']].plot.line(figsize=(30,10))
df[['HOXFLATMLM','HOXHOUSEMLM']].plot.line(figsize=(30,10))
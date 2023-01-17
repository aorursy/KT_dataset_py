# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/NOAA_UnitedStates2018.csv",delimiter=",",na_filter=False,low_memory=False)

df = df[df["REPORT_TYPE"] !='SOD  '] #remove the end-start of day records

df = df[df["HourlyDewPointTemperature"] !=''] #remove records that lack Dew Point data

subdf=df[['STATION','DATE',"HourlyDewPointTemperature"]] #creating a subdataframe

subdf.reset_index(drop=True) #drop the blank records & reindex

subdf = subdf[subdf["HourlyDewPointTemperature"] !='*'] #remove the records with * in the Dew Point

subdf = subdf[subdf["HourlyDewPointTemperature"] !='\s'] #remove records with an s

subdf = subdf[subdf["HourlyDewPointTemperature"] !='^(?!s$).*'] #remove records with an s in the middle (maybe?)

subdf = subdf[subdf["HourlyDewPointTemperature"] !='s$'] #remove records with an s at the end

subdf.dropna() #removes records with N/A (Not Applicable) data

subdf=subdf[~subdf.HourlyDewPointTemperature.str.contains("s")] #remove records containing an s

subdf=subdf[~subdf.HourlyDewPointTemperature.str.contains("HourlyDewPointTemperature")] #remove header artifacts from merging 11 CSV files

subdf.reset_index(drop=True) #resets the index, removes gaps in index numbering

subdf=subdf.apply(pd.to_numeric, errors='ignore') #converts STATION and HourlyDewPointTemperature to numbers (from objects-texts), so stats can be run

subdf.groupby('STATION').describe() #runs the statistics
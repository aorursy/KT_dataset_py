import os
org_wd=os.getcwd()
os.chdir('../input/aqi-helpers')
os.getcwd()
# import helpers
import read_aqi as rd
from my_helpers import check_missing_values
# changing working directory back
os.chdir(org_wd)
os.getcwd()
import numpy as np
import pandas as pd
aqi_2014=rd.read_aqi("../input/epa-erdb-2014/")
aqi_2015=rd.read_aqi("../input/epa-erdb-2015/")
aqi_2016=rd.read_aqi("../input/epa-erdb-2016/")
aqi_2017=rd.read_aqi("../input/epa-erdb-2017/")
aqi_20180106=rd.read_aqi("../input/epa-erdb-20180106/")
def unitDrop(DF, year):
    DF.drop(["SO2_UNIT","CO_UNIT","O3_UNIT",
            "PM10_UNIT","PM25_UNIT","NO2_UNIT",
            "WINDSPEED_UNIT","WINDDIREC_UNIT","ISCONVERTED",
            "FPMI","UNIT","NO_VALUE","DATACREATIONDATE",
            "TRANSACTIONID","FID"],
            axis=1,
            inplace=True)
    print("Success","dropping", year, sep=' ')
unitDrop(aqi_2014,2014)
unitDrop(aqi_2015,2015)
unitDrop(aqi_2016,2016)
unitDrop(aqi_2017,2017)
unitDrop(aqi_20180106,2018)
len(aqi_2014)
aqi_2014.head(10)
len(aqi_2015)
len(aqi_2016)
len(aqi_2017)
len(aqi_20180106)
from datetime import timedelta
# fix time format
def timetodt(TIME):
    try:
        try:
            return pd.to_datetime(TIME, format='%d-%m月-%y %I.%M.%S.%f000 上午')
        except ValueError:
            return pd.to_datetime(TIME, format='%d-%m月-%y %I.%M.%S.%f000 下午') + timedelta(hours=12)
    except ValueError:
        try:
            return pd.to_datetime(TIME, format='%d-%m月 -%y %I.%M.%S.%f000 上午')
        except ValueError:
            return pd.to_datetime(TIME, format='%d-%m月 -%y %I.%M.%S.%f000 下午') + timedelta(hours=12)
        # print(TIME,"is not a right format",sep=' ')
# takes a like a minute or two to process
# X['unigrams'] = X['text'].apply(lambda a: dmh.tokenize_text(a))
# # X['unigrams'] = X['text'].apply(dmh.tokenize_text)
# SO FREAKING FAST
def time_fix(TABLE):
    TABLE['UPDATETIME'] = TABLE['UPDATETIME'].apply(timetodt)
time_fix(aqi_2014)
time_fix(aqi_2015)
time_fix(aqi_2016)
time_fix(aqi_2017)
time_fix(aqi_20180106)


# count missing values in each column using similar helpers from lab 1
aqi_2014.isnull().apply(lambda x: check_missing_values(x))
aqi_2015.isnull().apply(lambda x: check_missing_values(x))
aqi_2016.isnull().apply(lambda x: check_missing_values(x))
aqi_2017.isnull().apply(lambda x: check_missing_values(x))
aqi_20180106.isnull().apply(lambda x: check_missing_values(x))
## save to pickle file
aqi_2014.to_pickle("aqi_tw_2014.pkl") 
aqi_2015.to_pickle("aqi_tw_2015.pkl")
aqi_2016.to_pickle("aqi_tw_2016.pkl")
aqi_2017.to_pickle("aqi_tw_2017.pkl")
aqi_20180106.to_pickle("aqi_tw_20180106.pkl")
## save to csv file
aqi_2014.to_csv("aqi_tw_2014.csv",index=False) 
aqi_2015.to_csv("aqi_tw_2015.csv",index=False)
aqi_2016.to_csv("aqi_tw_2016.csv",index=False)
aqi_2017.to_csv("aqi_tw_2017.csv",index=False)
aqi_20180106.to_csv("aqi_tw_20180106.csv",index=False)
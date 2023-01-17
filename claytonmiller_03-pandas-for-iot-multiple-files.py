import pandas as pd
buildingname = "Office_Abbey"
rawdata = pd.read_csv("../input/"+buildingname+".csv", parse_dates=True, index_col='timestamp')
rawdata.info()
rawdata.head()
rawdata.plot(figsize=(20,8))
meta = pd.read_csv("../input/all_buildings_meta_data.csv",index_col='uid')
meta.head()
meta.loc[buildingname]
meta.loc[buildingname]["sqm"]
rawdata.head()
rawdata_normalized = rawdata/meta.loc[buildingname]["sqm"]
rawdata_normalized.head()
rawdata_normalized_monthly = rawdata_normalized.resample("M").sum()
rawdata_normalized_monthly
rawdata_normalized_monthly.plot(kind="bar", figsize=(10,4))
rawdata_normalized_monthly.sum().plot(kind="bar", figsize=(5,4))
rawdata_normalized_monthly.index = rawdata_normalized_monthly.index.strftime('%b')
rawdata_normalized_monthly.plot(kind="bar", figsize=(10,4))
buildingnamelist = ["Office_Abbey",
"Office_Pam",
"Office_Penny",
"UnivLab_Allison",
"UnivLab_Audra",
"UnivLab_Ciel"]
annual_data_list = []
annual_data_list_normalized = []
all_data_list = []
for buildingname in buildingnamelist:
    print("Get the data from: "+buildingname)
    
    rawdata = pd.read_csv("../input/"+buildingname+".csv", parse_dates=True, index_col='timestamp')
    rawdata = rawdata[~rawdata.index.duplicated(keep='first')]
    
    all_data_list.append(rawdata[buildingname])

all_data = pd.concat(all_data_list, axis=1)
all_data.info()
all_data.head()
all_data.plot(figsize=(20,15), subplots=True)
all_data.resample("D").sum().plot(figsize=(20,15), subplots=True)
all_data.truncate(before='2015-02-01',after='2015-03-05').plot(figsize=(20,15), subplots=True)
all_data.truncate(before='2015-02-01',after='2015-02-05').plot(figsize=(20,15), subplots=True)
for buildingname in buildingnamelist:
    print("Getting data from: "+buildingname)
    
    rawdata = pd.read_csv("../input/"+buildingname+".csv", parse_dates=True, index_col='timestamp')
    floor_area = meta.loc[buildingname]["sqm"]
    
    annual = rawdata.sum()

    normalized_data = rawdata/floor_area
    annual_normalized = normalized_data.sum()
    
    annual_data_list_normalized.append(annual_normalized)
    annual_data_list.append(annual) 
totaldata = pd.concat(annual_data_list)
totaldata_normalized = pd.concat(annual_data_list_normalized)
totaldata
totaldata_normalized
totaldata.plot(kind='bar',figsize=(10,5))
totaldata_normalized.plot(kind='bar',figsize=(10,5))

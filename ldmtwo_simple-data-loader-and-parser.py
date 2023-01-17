#credits to original: https://www.kaggle.com/scirpus/to-overcome-the-terrible-data
#Please thank him above ^ as this was much simpler thanks to Panda's json_normalize
#key changes to his version: removed joins, inplace drop, and made it more verbose
import pandas as pd
import json,os,gc,time
def currenttime(): return time.time()
def Load(file, max_rows=None, DROP=[], with_prefix=True):
    from pandas.io.json import json_normalize
    t0 = currenttime()
    columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(file, 
                     converters={column: json.loads for column in columns}, 
                     dtype={'fullVisitorId': 'str'}, 
                     nrows=max_rows)
    print(df.shape)
    lst = [df]
    for c in columns:
        tmp = json_normalize(df[c])
        if with_prefix: tmp.columns = [f"{c}.{subcolumn}" for subcolumn in tmp.columns]
        lst.append(tmp)     
        print('%20s  +%s'%(c,tmp.shape))
    df = pd.concat(lst, axis=1)
    print('->',df.shape)
    DROP = [c for c in columns+DROP if c in df.columns]
    df.drop(DROP, axis=1, inplace=True)
    print('DROP %s ->'%len(DROP),df.shape)
    print('Loaded %s in %s sec'%(file,currenttime()-t0))
    return df

#feel free to check for your self (set it to [])
useless=['device.browserSize', 'device.browserVersion', 'device.flashVersion', 'device.language', 'device.mobileDeviceBranding', 'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName', 'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.operatingSystemVersion', 'device.screenColors', 'device.screenResolution', 'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.networkLocation', 'socialEngagementType', 'totals.visits', 'trafficSource.adwordsClickInfo.criteriaParameters','trafficSource.campaignCode']

#full data size will be (1708337, 36)
Load('../input/train.csv',max_rows=5000,DROP=useless).sample(5)
def GetData():
    R = Load('../input/train.csv',DROP=useless)
    E = Load('../input/test.csv',DROP=useless)
    R['is_train'] = 1
    E['is_train'] = 0
    print('Train/test dimensions:',R.shape,E.shape)
    return R.append(E, sort=True)

data = GetData()
gc.enable()
gc.collect()
print()
S = pd.read_csv('../input/sample_submission.csv')
print()
print(sorted(data.columns),'\n')
print(sorted(S.columns),'\n')
data.shape,S.shape
def H5DF_load(fileName):
    t0 = currenttime()
    if not os.path.exists(fileName):
        raise ValueError("Error: file does not exits: {}".format(fileName))
    try:
        print("Reading %20s" % fileName, end='', flush=True)
        data = pd.read_hdf(fileName,'table')
        print(" %15s  %.3f seconds" % (data.shape, currenttime()-t0))
        return data
    except:
        raise ValueError("file = {}".format(fileName))

#Expects a dataframe
def H5DF_save(data,fileName, append=False):
    t0 = currenttime()
    print("Saving %20s" % fileName, end='', flush=True)
    data.to_hdf(fileName,'table',append=append)
    print(" %15s  %.3f seconds" % (data.shape, currenttime()-t0))

#if you plan to reload this often    
H5DF_save(data,'data.h5df')
data = H5DF_load('data.h5df')

# The warning given is because of irregular data types. It results to a more complicated serialization process. To resolve, you need to set the data type appropriately. It will be much faster!
data.sample(100)
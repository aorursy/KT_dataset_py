import numpy as np 
import pandas as pd 
import json
import pandas.io.json as pdjson

import ast

import os
print(os.listdir("../input"))
#path = 'C:\\Users\\Andre\\code\\Kaggle\\2. Google Analytics Customer Revenue Prediction\\train.csv'
path = '../input/train_v2.csv'
data1 = pd.read_csv(path, sep=',', dtype={'fullVisitorId': 'str'}, nrows=100)
del(path)

# load the competition test data
#path = 'C:\\Users\\Andre\\code\\Kaggle\\2. Google Analytics Customer Revenue Prediction\\test.csv'
path = '../input/test_v2.csv'
data2 = pd.read_csv(path, sep=',', dtype={'fullVisitorId': 'str'}, nrows=100)
del(path)

print('data1 shape:', data1.shape)
data1.head()
print('data2 shape:', data2.shape)
data2.head()
data = data1.append(data2, ignore_index=True)
del(data1, data2)
print(' data shape:', data.shape)
data.customDimensions[0]
data.device[0]
data.geoNetwork[0]
data.hits[0]
data.totals[0]
data.trafficSource[0]
# Any results you write to the current directory are saved as output.

def parse(csv_path, nrows=None):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    device_list=df['device'].tolist()
    
    #deleting unwanted columns before normalizing
    for device in device_list:
        del device['browserVersion'],device['browserSize'],device['flashVersion'],device['mobileInputSelector'],device['operatingSystemVersion'],device['screenResolution'],device['screenColors']
    df['device']=pd.Series(device_list)
    
    geoNetwork_list=df['geoNetwork'].tolist()
    for network in geoNetwork_list:
        del network['latitude'],network['longitude'],network['networkLocation'],network['cityId']
    df['geoNetwork']=pd.Series(geoNetwork_list)
    
    df['hits']=df['hits'].apply(ast.literal_eval)
    df['hits']=df['hits'].str[0]
    df['hits']=df['hits'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    
    df['customDimensions']=df['customDimensions'].apply(ast.literal_eval)
    df['customDimensions']=df['customDimensions'].str[0]
    df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)
    
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource','hits','customDimensions']

    for column in JSON_COLUMNS:
        column_as_df = pdjson.json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    
    return df
print("The 'parse' function to flatten JSON columns have been created")
data1 = parse('../input/train_v2.csv', nrows=100000)
data2 = parse("../input/test_v2.csv",nrows=100000)

print('data1 shape: ', data1.shape)
print('data2 shape: ', data2.shape)

data = data1.append(data2, sort=True)
del(data1, data2)

print('number of unique columns in data1 + data2:', data.shape)
jsonlist=[]
for i in range(len(data.columns)):   # for each column
    if (isinstance(data.iloc[1,i], list) ):  # see if some element 1 is a list
        jsonlist.append( data.columns[i] )   # if yes, then save name to list
print(jsonlist)
print("Printout for each column's number of unique values (incl. nans)\n")
for col in data.columns:
    try:
        print(col, ':', data[col].nunique(dropna=False))
    except TypeError:
        a=data[col].astype('str')
        #print(a)
        print( col, ':', a.nunique(dropna=False), ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LIST')
# Clean workspace
del(col)
print('Data shape before dropping constant columns:', data.shape)

print('\nColumns being dropped:')

for col in data.columns:
    try:
        if (data[col].nunique(dropna=False) == 1):
            del(data[col])
            print(col)
    except TypeError:
        a=data[col].astype('str')
        if (a.nunique(dropna=False) == 1):
            del(data[col])
            print(col)
del(col)

print('\ndata shape is now:', data.shape)
data.head()
print("Printout for each column's number of unique values (incl. nans)\n")
for col in data.columns:
    try:
        print(col, ':', data[col].nunique(dropna=False))
    except TypeError:
        a=data[col].astype('str')
        #print(a)
        print( col, ':', a.nunique(dropna=False), ' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LIST')
# Clean workspace
del(col)
print('number of unique values in column:', 
      data['hits_product'].astype('str').nunique(dropna=False), '\n' )
print( data['hits_product'].iloc[0] )
print('data shape:', data.shape)
data = data.drop(labels=['hits_product'], axis=1)
print('Removed hits_product')
print('data shape:', data.shape)
print('number of unique values in column:', 
      data['hits_promotion'].astype('str').nunique(dropna=False), '\n' )
print( data['hits_promotion'].iloc[1] )
print('data shape:', data.shape)
data = data.drop(labels=['hits_promotion'], axis=1)
print('Removed hits_promotion')
print('data shape:', data.shape)
print("Printout for each column's number of unique values (incl. nans)\n")
for col in data.columns:
    try:
        print(col)
    except TypeError:
        a=data[col].astype('str')
        #print(a)
        print( col)
# Clean workspace
del(col)
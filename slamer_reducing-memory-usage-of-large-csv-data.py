# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import glob
listBigCSV = glob.glob("../input/NGS*csv")
listBigCSV
# Any results you write to the current directory are saved as output.
def memory(df):
    if isinstance(df,pd.DataFrame):
        value = df.memory_usage(deep=True).sum() / 1024 ** 2
    else: # we assume if not a df it's a series
        value = df.memory_usage(deep=True) / 1024 ** 2
    return value, "{:03.2f} MB".format(value)
df = pd.read_csv(listBigCSV[2], engine='c')
df.describe()
df.dtypes
dfIntSelection = df.select_dtypes(include=['int'])
dfConverted2int = dfIntSelection.apply(pd.to_numeric,downcast='unsigned')
memInt, memIntTxt=  memory(dfIntSelection)
memIntDownCast, memIntDownCastTxt = memory(dfConverted2int)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)
dfConverted2int.describe()
dfIntSelection = df.select_dtypes(include=['int'])
dfConverted2int = dfIntSelection.astype('category')
memInt, memIntTxt=  memory(dfIntSelection)
memIntDownCast, memIntDownCastTxt = memory(dfConverted2int)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)

dfFloatSelection = df.select_dtypes(include=['float'])
dfConverted2float = dfFloatSelection.apply(pd.to_numeric,downcast='float')
memInt, memIntTxt=  memory(dfFloatSelection)
memIntDownCast, memIntDownCastTxt = memory(dfConverted2float)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)
dfTime = df.Time 
date_format = '%Y-%m-%d %H:%M:%S.%f'
dfTimeConvert = pd.to_datetime(dfTimeConvert,format=date_format)

mem, memTxt = memory(dfTime)
memConv, memConvTxt = memory(dfTimeConvert)

print(memTxt)
print(memConvTxt)
print('Gain: ', mem/memConv *100.0)
dtypes = df.drop('Time',axis=1).dtypes

dtypes_col = dtypes.index
dtypes_type = [i.name for i in dtypes.values]

column_types = dict(zip(dtypes_col, dtypes_type))
preview = first2pairs = {key:value for key,value in list(column_types.items())[:10]}
import pprint
pp = pp = pprint.PrettyPrinter(indent=4)
pp.pprint(preview)
dfDownCast =pd.read_csv(listBigCSV[2],dtype=column_types,parse_dates=['Time'], infer_datetime_format=True)

dfDownCast
memInt, memIntTxt=  memory(df)
memIntDownCast, memIntDownCastTxt = memory(dfDownCast)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)
dfDownCast.info()

def downCast(df):
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    converted_obj = df.select_dtypes(include=['int']).astype('category')
    df[converted_obj.columns] = converted_obj
    converted_obj = df.select_dtypes(include=['float']).apply(pd.to_numeric,downcast='float')
    df[converted_obj.columns] = converted_obj
    if 'Time' in df:
        df.Time = pd.to_datetime(df.Time,format=date_format)
    return df
dfDown = df.copy()
dfDown = downCast(dfDown)

memInt, memIntTxt=  memory(df)
memIntDownCast, memIntDownCastTxt = memory(dfDown)

print(memIntTxt)
print(memIntDownCastTxt)
print('Gain: ', memInt/memIntDownCast *100.0)
for csvFile in listBigCSV:
    dataframe = pd.read_csv(csvFile, engine='c')
    dataframe = downCast(dataframe)
    dataframe.to_pickle(os.path.basename(csvFile[:-4]+'.pkl'))
    del dataframe
os.path.basename('../input/NGS-2016-reg-wk7-12.pkl')
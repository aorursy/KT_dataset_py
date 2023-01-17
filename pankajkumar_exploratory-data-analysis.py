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
tp = pd.read_csv('../input/train.csv', iterator=True, chunksize=1000000000)
df1 = pd.concat(tp, ignore_index=True)
df1.describe()
header = ['Variable','Data_Type','Observation', 'Missing','Unique_Value','Mean','STD','Min','Pct25','Pct50','Pct75','Pct99','Max','Mode']
temp1 = pd.DataFrame(np.array([np.arange(1)]*14).T,columns=header)
temp = pd.DataFrame(np.array([np.arange(1)]*14).T,columns=header)
for var in df1:
    if (df1[var].dtype.name == 'int64') or (df1[var].dtype.name == 'float64'): # EDA for all numeric type variable
        summary = pd.DataFrame(df1[var].describe()).reset_index().\
                    rename(index=str, columns={var: "value"})    
        Pct99 = pd.DataFrame(df1[var].quantile([.99]))    
        temp1[['Variable']] = var
        temp1[['Data_Type']] = df1[var].dtype.name
        temp1[['Observation']] = int(len(df1))
        temp1[['Missing']] = df1[var].isnull().values.ravel().sum()
        temp1[['Unique_Value']] = df1[var].nunique()
        temp1[['Mean']] = summary.iloc[1]['value']
        temp1[['STD']] = summary.iloc[2]['value']
        temp1[['Min']] = summary.iloc[3]['value']
        temp1[['Pct25']] = summary.iloc[4]['value']
        temp1[['Pct50']] = summary.iloc[5]['value']
        temp1[['Pct75']] = summary.iloc[6]['value']
        temp1[['Pct99']] = Pct99.iloc[0][0]
        temp1[['Max']] = summary.iloc[7]['value']
        temp1[['Mode']] = pd.DataFrame(df1[var].mode()).iloc[0] # sometime two mode value is avialable 
        temp = temp.append(temp1, ignore_index=True)
    else: # EDA for String Variable
        summary1 = pd.DataFrame(df1[var].describe()).reset_index().\
                   rename(index=str, columns={var: "value"})      
        temp1[['Variable']] = var
        temp1[['Data_Type']] = df1[var].dtype.name
        temp1[['Observation']] = int(len(df1))
        temp1[['Missing']] = df1[var].isnull().values.ravel().sum()
        temp1[['Unique_Value']] = df1[var].nunique()
        temp1[['Mean']] = np.nan
        temp1[['STD']] = np.nan
        temp1[['Min']] = np.nan
        temp1[['Pct25']] = np.nan
        temp1[['Pct50']] = np.nan
        temp1[['Pct75']] = np.nan
        temp1[['Pct99']] = np.nan
        temp1[['Max']] = np.nan
        temp1[['Mode']] = pd.DataFrame(df1[var].mode()).iloc[0] # sometime two mode value is avialable 
        temp = temp.append(temp1, ignore_index=True)
print('Out of loop')
temp.drop(temp.index[0], inplace=True) 
EDA_Output = temp;
EDA_Output.to_csv('EDA_Output.csv',index=False, sep='\t', encoding = 'utf-8')
EDA_Output
import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import datetime

from datetime import date
# Read data

data= pd.read_csv('../input/lannettechtask1/date_data.csv', index_col =0, parse_dates=['DATE1'])
data
# Check for nulls

data.isnull().sum().sum()
data.DATE1.dtype
# Check for invalid dates, if any

data['DATE1'].map(lambda x: x.split('-')[0]).sort_values().unique()
# Check for invalid months, if any

data['DATE1'].map(lambda x: x.split('-')[1]).sort_values().unique()
# Check for invalid year, if any

data['DATE1'].map(lambda x: x.split('-')[2]).sort_values().unique()
# Replacing all invalid days with 30

for x in filter(lambda w: w > 31 , data['DATE1'].map(lambda x: int(x.split('-')[0] ))):

      data['DATE1'].replace(to_replace = str(x), value = '30', regex = True, inplace = True) 
data['DATE1'].map(lambda x: int(x.split('-')[0] )).sort_values().unique()
# Replacing all invalid months with 12

for x in filter(lambda w: w > 12 , data['DATE1'].map(lambda x: int(x.split('-')[1] ))):

      data['DATE1'].replace(to_replace = str(x), value = '12', regex = True , inplace = True) 
data['DATE1'].map(lambda x: int(x.split('-')[1] )).sort_values().unique()
data['DATE1'] = pd.to_datetime(data['DATE1'], infer_datetime_format = True, dayfirst=True)
# Check datatype of DATE1

data.DATE1.dtype
data.head()
data.DATE2.dtype
# Check for invalid months, if any

data['DATE2'].map(lambda x: x.split(',')[1].strip().split(' ')[0]).unique()
# Check for invalid dates, if any

data['DATE2'].map(lambda x: x.split(',')[1].strip().split(' ')[1]).unique()
# Check for invalid year, if any

data['DATE2'].map(lambda x: x.split(',')[2].strip()).unique()
# Correct the spelling of December

data.DATE2.replace('Desember','December',regex = True , inplace =True)
data['DATE2'].map(lambda x: x.split(',')[1].strip().split(' ')[0]).unique()
# Replacing all invalid days with 30

for x in filter(lambda w: w > 31 , data['DATE2'].map(lambda x: int(x.split(',')[1].strip().split(' ')[1]))):

      data['DATE2'].replace(to_replace = str(x), value = '30', regex = True, inplace = True) 
data['DATE2'].map(lambda x: int(x.split(',')[1].strip().split(' ')[1])).sort_values().unique()
data['DATE2'] = pd.to_datetime(data['DATE2'], format ="%A, %B %d, %Y")
# Check datatype of DATE2

data.DATE2.dtype
data.head()
data['DATE3'].map(lambda x: x.split('-')[0]).sort_values().unique()
data['DATE3'].map(lambda x: x.split('-')[1]).sort_values().unique()
data['DATE3'].map(lambda x: x.split('-')[2]).sort_values().unique()
data['DATE3'] = pd.to_datetime(data['DATE3'], format ="%d-%b-%y")
# Checking datatyoe of DATE3

data.DATE3.dtype
data.head()
data['DATE4'].map(lambda x: x.split('-')[0]).sort_values().unique()
data['DATE4'].map(lambda x: x.split('-')[1]).sort_values().unique()
data['DATE4'].map(lambda x: x.split('-')[2]).sort_values().unique()
data['DATE4'] = pd.to_datetime(data['DATE4'], format ="%d-%b-%y")
data.head()
base_date = pd.Timestamp('1899-12-30')

data.DATE5= data['DATE5'].map(lambda x: base_date + pd.DateOffset(x))
# Checking datatype of DATE5

data.DATE5.dtype
# Check for nulls

data.isnull().sum().sum()
data.head()
def date_diff(df):

  for i in range(len(df.columns)):

   for j in range(i+1,len(df.columns)):

     if (df.dtypes[i] == '<M8[ns]') and (df.dtypes[j] == '<M8[ns]'):

      df['{}-{}'.format(df.columns[i],df.columns[j])]= data[data.columns[i]] -data[data.columns[j]]
date_diff(data)
data.head()
# Read data and check for nulls

data_1 = pd.read_csv('../input/lannettechtask2/1000 Sales Records.csv')

data_1.isnull().sum().sum()
data_1.head()
data_1.shape
def drop_pc(df):

  # Create correlation matrix

  corr_matrix = df.corr(method = 'pearson').abs()

  # Select upper triangle of correlation matrix

  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

  # Return list of columns with correlation coefficient greater than 0.85

  return [column for column in upper.columns if any(upper[column] > 0.85)] 
# Print columns that will be dropped

drop_pc(data_1)
# Drop columns with correlation coefficient greater than 0.85

data_1.drop(drop_pc(data_1), axis=1)
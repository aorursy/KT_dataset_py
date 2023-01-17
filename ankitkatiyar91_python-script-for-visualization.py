# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime as dt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
all_files = os.listdir("../input")

# Any results you write to the current directory are saved as output.
csv_files = [f for f in all_files if f.endswith('.csv')][0] 
print(csv_files)
df = pd.read_csv('../input/'+csv_files)
df.head()
df['StartDateActual'] = [d.date() for d in pd.to_datetime(df['StartDateActual'])]
df['StartDateActual'].head()
print ('Oldest Project started at: ' + min([d for d in df['StartDateActual'] if pd.notnull(d)]).strftime("%d-%b-%Y"))
print ('Latest Project started at: ' + max([d for d in df['StartDateActual'] if pd.notnull(d)]).strftime("%d-%b-%Y"))
worst_proj_idx = df['CO2e (MT) Calculated'].sort_values(axis =0,ascending =False)[1:11]
# Get Index out of pandas series and convert to list
#worst_proj_idx.index.tolist()
df.iloc[worst_proj_idx.index.tolist(),:10]
status_group = df.groupby('ProjectStatus')
status_group.size().reset_index(name='counts')
#df['CO2e (MT) Calculated']
df['FinishDateActual'] = [d.date() for d in pd.to_datetime(df['FinishDateActual'])]
df['Days']=df['FinishDateActual']-df['StartDateActual']
#print(df['Days'])
df['Days'] = list(map( lambda x: x.days,df['Days']))
df['Efficiency'] = df['CO2e (MT) Calculated']/df['Days']
print(df['Efficiency'].head())
print("Minimum:",min([d for d in df['Efficiency'] if pd.notnull(d)]))
print("Maximum:",max([d for d in df['Efficiency'] if pd.notnull(d)]))
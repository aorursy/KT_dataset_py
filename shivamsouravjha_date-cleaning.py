# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


mf = pd.read_csv('../input/nypd-motor-vehicle-collisions/nypd-motor-vehicle-collisions.csv')


def getdates(df):
  

  for col in df.columns:
    if df[col].dtype == 'object':
        try:
              df[col] = pd.to_datetime(df[col],format = '%Y%m%d',utc=True,errors='ignore',infer_datetime_format=True)
        except :
          try:
              df[col] = pd.to_datetime(df[col],format = '%Y/%m/%d',infer_datetime_format=True)
          except:
              try:
                  df[col] = pd.to_datetime(df[col],format = '%Y-%m-%d',infer_datetime_format=True)
              except:
                  try:
                    df[col] = pd.to_datetime(df[col],format = '%d/%m/%Y',infer_datetime_format=True)
                  except:
                    df[col] = pd.to_datetime(df[col],format = '%Y%m%d',infer_datetime_format=True)

  for col in df.columns:
  
    if df[col].dtype == 'object'  or   df[col].dtype == 'float64'  or df[col].dtype ==  'int64' :
      try:
        df.drop([col],axis= 1,inplace=True)
      except:
        print(1)
mf.head()
def getdifference(df):
  getdates(df)
getdifference(mf)
 

from itertools import combinations
columns  = mf.columns

for col in combinations(columns,2):
          try:
              name = str(col[0]) + '_' + str(col[1])
              mf[name] = mf[col[0]] - mf[col[1]]
          except ValueError:
              print(1)
              pass


mf.head(8)
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
d_10101 = pd.read_csv('../input/10101.csv')
d_10001 = pd.read_csv('../input/10001.csv')
d_10003 = pd.read_csv('../input/10003.csv')
d_Uptime = pd.read_csv('../input/Uptime.csv')
d_10101_onlymax=d_10101.groupby(['CPE Id'], sort=False)['Value'].max().to_frame()
d_10001_onlymax=d_10001.groupby(['CPE Id'], sort=False)['Value'].max().to_frame()
d_10003_onlymax=d_10003.groupby(['CPE Id'], sort=False)['Value'].max().to_frame()
d_Uptime_onlymax=d_Uptime.groupby(['CPE Id'], sort=False)['Value'].max().to_frame()
d_10101_onlymax.head()
d_10003_onlymax.head()
d_10101_10001=d_10101_onlymax.merge(d_10001_onlymax,on='CPE Id',how='outer',suffixes=('_10101', '_10001')).fillna(0)
d_10101_10001.head()
d_10101_10001_10003=d_10101_10001.merge(d_10003_onlymax,on='CPE Id',how='outer').fillna(0)    
d_10101_10001_10003.rename(columns={'Value':'Value_10003'},inplace=True)
d_10101_10001_10003.head(100)
result=pd.merge(d_Uptime_onlymax,d_10101_10001_10003,on='CPE Id',how='left').fillna(0)
result.rename(columns={'Value':'Uptime'},inplace=True)
result.head(5)
result["Sum"]=result["Value_10101"] + result["Value_10001"] +result["Value_10003"]
result.head()
result.to_csv("result.csv")
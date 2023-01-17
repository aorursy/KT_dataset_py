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
colName = ["TimeInterval", "GantryID", "Direction", "VehicleType" , "Counts"]
def read_traffic(DIRNAME):
    output = pd.DataFrame()
    print("Start importing files from", DIRNAME, sep=' ',end='\n')
    for root, dirs, files in os.walk(DIRNAME, topdown = False):
        for name in files:
            oldRead = output
            newRead = pd.read_csv(os.path.join(root, name), header=None)
            output = pd.concat([oldRead, newRead], ignore_index=True)
            print("Reading",os.path.join(root, name),sep=' ',end='\n')
    print("Done importing files from", DIRNAME, sep=' ',end='\n')
    output.columns = colName
    return output
traffic_=read_traffic("../input")
traffic_.head(10)
from datetime import datetime
# fix time format
def timetodt(TIME):
    return pd.to_datetime(TIME, format='%Y-%m-%d %H:%M')

def time_fix(TABLE):
    TABLE['TimeInterval'] = TABLE['TimeInterval'].apply(timetodt)

time_fix(traffic_)
traffic_.to_csv("output.csv",index=False)
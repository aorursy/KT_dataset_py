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
import pandas as pd                        
import json
with open("Covid_Tweets_Kaggle.json") as df:
    df = df.readlines()
#%%
testDF = pd.DataFrame()
rf = json.loads(df[3222])
columns = []
for key, value in rf.items():
    print (key, value)
    columns.append(key)
    testDF[key] = ""
#%%
for i in range(0, len(df)):
    if len(df[i])>2:
        rf = json.loads(df[i])
        df_dat= {}
        for key, value in rf.items():
            #print(key,value)
            vsl = []
            vsl.append(value)
            df_dat[key] = vsl            
        df_data = pd.DataFrame(df_dat)
        testDF = pd.concat([testDF,df_data])
#%%
with open("Covid_Tweets_Kaggle1.json") as df:
    df = df.readlines()
for i in range(0, len(df)):
    if len(df[i])>2:
        rf = json.loads(df[i])
        df_dat= {}
        for key, value in rf.items():
            #print(key,value)
            vsl = []
            vsl.append(value)
            df_dat[key] = vsl            
        df_data = pd.DataFrame(df_dat)
        testDF = pd.concat([testDF,df_data])    
#%%
with open("Covid_Tweets_Kaggle2.json") as df:
    df = df.readlines()
for i in range(0, len(df)):
    if len(df[i])>2:
        rf = json.loads(df[i])
        df_dat= {}
        for key, value in rf.items():
            #print(key,value)
            vsl = []
            vsl.append(value)
            df_dat[key] = vsl            
        df_data = pd.DataFrame(df_dat)
        testDF = pd.concat([testDF,df_data])       
testDF.to_csv("final.csv")
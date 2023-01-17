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
#1.)Extracion of data('Company Name','Location','Datum','Detail','Status Rocket',' Rocket','Status Mission')
import pandas as pd
df=pd.read_csv("/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv")
df1=df[['Company Name','Location','Datum','Detail','Status Rocket',' Rocket','Status Mission']]
df1
#2.) Counting Successful Missions
count=len(df1[df1['Status Mission'] == 'Success'])
print("Total number of successful missions are:",count)
#3.) Counting Failed Missions
count1=len(df1[df1['Status Mission'] == 'Failure'])
print("Total number of Failed Missions are:",count1)
#4.) Different Rocket Status
df2=df1.groupby(' Rocket').count()
df2
#5.)Counting How many belonged to USA.
count=len(df1)
df3=pd.DataFrame()
for i in range(0,count):
    #checking for USA and adding data to new dataframe
    location=df1.loc[i,['Location']]
    split=location.str.split(", ")
    new_list = []
    for item in split:
        new_list.append((item))
    b=new_list[0]
    if 'USA'in b:
        df3 = df3.append(df1.iloc[i])
    else:
        continue
   
df3
    

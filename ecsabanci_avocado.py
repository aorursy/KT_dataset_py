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
df = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")

df["Date"] = pd.to_datetime(df["Date"])

albany_df = df[df["region"] == "Albany"]
albany_df.set_index("Date",inplace = True)
albany_df["AveragePrice"].plot()
albany_df["AveragePrice"].rolling(25).mean().plot()
albany_df.sort_index(inplace = True)
albany_df["AveragePrice"].rolling(25).mean().plot()
albany_df["price25ma"] = albany_df["AveragePrice"].rolling(25).mean()
albany_df.head(3)
albany_df.dropna().head(3)
albany_df = df.copy()[df["region"] == "Albany"]
albany_df.set_index("Date", inplace = True)
albany_df.sort_index(inplace = True)
albany_df["prime25ma"] = albany_df["AveragePrice"].rolling(25).mean()
list(set(df["region"].values.tolist()))
df["region"].unique()
graph_df = pd.DataFrame()

for region in df["region"].unique()[:16]:
    print(region)
    region_df = df.copy()[df["region"] == region]
    region_df.set_index("Date", inplace = True)
    region_df.sort_index(inplace = True)
    region_df[f'{region}_prime25ma'] = region_df["AveragePrice"].rolling(25).mean()
    
    if graph_df.empty:
        grap_df = region_df[[f'{region}_prime25ma']]
        
    else:
        graph_df = graph_df.join(region_df[f'{region}_prime25ma'])

#None of [Index(['Albanyprice25ma'], dtype='object')] are in the [columns]
df["type"].unique()
df = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")

df = df.copy()[df['type'] == "organic"]
df['Date'] = pd.to_datetime(df["Date"])

df.sort_values(by="Date", ascending = True, inplace = True)



graph_df = pd.DataFrame()

for region in df['region'].unique():
    print(region)
    region_df = df.copy()[df['region'] == region]
    region_df.set_index("Date", inplace = True)
    region_df.sort_index(inplace = True)
    region_df[f'{region}_prime25ma'] = region_df['AveragePrice'].rolling(25).mean()
    
    if graph_df.empty:
        graph_df = region_df[[f'{region}_prime25ma']]
        
    else:
        graph_df = graph_df.join(region_df[f'{region}_prime25ma'])
        
graph_df.tail()
        
        
graph_df.dropna().plot(figsize = (8,5), legend = False)

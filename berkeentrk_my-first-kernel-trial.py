# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pandas as pd 
from datetime import date,timedelta,datetime
import plotly.express as px

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv",parse_dates=True)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df.info()
df.isnull().sum()
df.head()
df.drop("Port Code,axis=1") #Deleting the Port Code column to make dataset more neat and efficient
#Converting datetime in pandas
df['Date'] = pd.to_datetime(df['Date'])
most_used =df.groupby("Measure")["Value"].sum().sort_values(ascending=False).reset_index()
fig = px.bar(most_used,
             x = "Measure",
             y = "Value" ,
             template = "plotly_dark")
fig.show()
gbb = pd.DataFrame(df.groupby(by="State")["Value"].sum().sort_values(ascending=False)).reset_index()
fig = px.pie(gbb,
             names="State",
             values="Value",      
             template="seaborn")
fig.show()
gbb =  df.groupby("Border")[["Value"]].sum().reset_index()
fig = px.pie(gbb,
             values="Value",
             names="Border",
             template="seaborn")

fig.show()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sqlite3
import warnings  
warnings.filterwarnings('ignore')
import plotly.plotly as py # interactive graphing
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

conn = sqlite3.connect(':memory:')
#print(os.listdir("../input"))
data = pd.read_csv("../input/procurement-notices.csv")
#data.info()
#data.head()
data.to_sql('wb', conn, if_exists='append', index=False)
c = conn.cursor()
sql = """select "Country Name" as Country, "Procurement Type" as ProcurementType, count("Procurement Type") as count from wb group by Country,ProcurementType order by count DESC;"""
qd = pd.read_sql_query(sql, conn)
qd.head(20)
#sql2 = """select "Notice Type", "Publication Date", "Country Name" from wb group by "Publication Date";"""
sql2 = """select "Publication Date", "Country Name",count("Country Name") as count from wb group by "Publication Date","Country Name" order by count DESC ;"""

qd2 = pd.read_sql_query(sql2,conn)
#qd2.head(20)
gdata = [go.Scatter(x=qd2["Publication Date"], y=qd2["count"],mode='markers',text=qd2["Country Name"])]
layout = dict(title = "Distribution of publications over time for countries",
              xaxis= dict(title= 'Publication Date',ticklen= 5,zeroline= False))

fig = dict(data = gdata, layout = layout)
iplot(fig)
sql3 = """select "Publication Date", "Procurement Type",count("Procurement Type") as count from wb group by "Publication Date","Procurement Type" order by count DESC ;"""

qd3 = pd.read_sql_query(sql3,conn)
#qd3.head(20)
gdata1 = [go.Scatter(x=qd3["Publication Date"], y=qd3["count"],mode='markers',text=qd3["Procurement Type"])]
layout1 = dict(title = "Distribution of procurementtypes over time",
              xaxis= dict(title= 'Publication Date',ticklen= 5,zeroline= False))

fig1 = dict(data = gdata1, layout = layout1)
iplot(fig1)
#sql4 = """select "Publication Date", "Procurement Type",count("Procurement Type") as count from wb group by "Publication Date","Procurement Type" order by count DESC ;"""
sql4 =  """select "Country Name" as Country, count("Country Name") as count from wb group by Country order by count DESC;"""
qd4 = pd.read_sql_query(sql4,conn)
#qd3.head(20)
gdata2 = [go.Bar(x=qd4["Country"], y=qd4["count"],orientation = 'v')]
layout2 = dict(title = "Drawers by number of bids overview")
fig2 = dict(data = gdata2, layout = layout2)
iplot(fig2)

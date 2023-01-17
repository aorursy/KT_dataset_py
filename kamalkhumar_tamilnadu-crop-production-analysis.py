# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd

import missingno as msno
import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/tamilnadu-cropproduction/Tamilnadu agriculture yield data.csv")
df.head()
df.info()
df.State_Name.value_counts()
df.drop("State_Name", axis = 1, inplace = True)
df.Crop_Year.value_counts().sort_index()
df.isna().sum()
msno.matrix(df)
plt.show()
df.dropna(how = "any", inplace = True)
print("No. of duplicate entries: ",len(df[df.duplicated()]))
df.District_Name = df.District_Name.apply(lambda x: x.lower().capitalize())
df.groupby("Crop_Year")["Area"].sum().sort_index(ascending = True)
real = pd.DataFrame({"Year": df.groupby("Crop_Year")["Area"].sum().sort_index(ascending = True).index,
                   "Total area": df.groupby("Crop_Year")["Area"].sum().sort_index(ascending = True).values})
real.head()
import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x = real['Year'], y = real['Total area'], marker_color = real['Total area']))
fig.update_layout(title='Agricultural area over the years',  xaxis = dict(tickmode = 'linear', dtick = 1))
fig.show()
df.groupby("District_Name")["Area"].sum().sort_values(ascending = False)
dis = pd.DataFrame({"District": df[df.Crop_Year == 2013].groupby("District_Name")["Area"].sum().sort_values(ascending = False).index,
                   "Total area": df[df.Crop_Year == 2013].groupby("District_Name")["Area"].sum().sort_values(ascending = False).values})
dis.head()
import plotly.express as px

fig = px.bar(dis, x='District', y='Total area', color='Total area', height = 500,width = 1100, text = 'Total area', title = "Agricultural area of each district in the year 2013")
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
df.Season.value_counts()
df.Crop.value_counts()
df.groupby(["Season", "Crop"])["Production"].sum()
sp = pd.DataFrame({'Production' : df.groupby(["Season", "Crop"])["Production"].sum()}).reset_index()
sp.head()
sp.Season.value_counts()
wy = sp[sp['Season']=='Whole Year']
el = sp[sp['Season']!='Whole Year']
sp = pd.concat([wy.sample(frac = 0.3), el]).sample(frac=1)
fig =px.sunburst(sp,path=['Season', 'Crop'], values='Production')
fig.show()
fig = px.scatter(df, x="Production", y="Area",size="Crop_Year", color="Season", log_x=True, size_max=10, title = "Area vs Production distribution")
fig.show()
ni = pd.DataFrame({'Production' : df.groupby(["District_Name", "Season"])["Production"].sum()}).reset_index()
ni = ni.sort_values("Production", ascending = False)
ni = ni[ni.Season != "Whole Year"]
ni.head()
fig = px.bar(ni, x = "District_Name", y = "Production", color='Season', title = "Kharif vs Rabi production distribution")
fig.show()
t = pd.DataFrame({"Production":df.groupby(["Season","District_Name","Crop"])["Production"].sum()}).reset_index()
t = t.sort_values("Production", ascending = False)
fig = go.Figure(data=[go.Table(
    header=dict(values=list(t.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[t.Season, t.District_Name, t.Crop, t.Production],
               fill_color='lavender',
               align='left'))
])

fig.show()
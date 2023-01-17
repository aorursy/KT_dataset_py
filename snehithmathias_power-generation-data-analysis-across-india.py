import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools
data1 = pd.read_csv("../input/daily-power-generation-in-india-20172020/State_Region_corrected.csv")
data1.head()
data1.info()

total_Area = data1["Area (km2)"].sum()
print(total_Area)
total_Area_p = (data1["Area (km2)"]/total_Area)*100
print(total_Area_p)


Pie = go.Pie(labels=data1["State / Union territory (UT)"], 
            values=total_Area_p)
layout = go.Layout(title="% Area by each State",
                  height=1000,
                  width=1000)
fig = go.Figure(data=Pie, layout=layout)
fig.show()
Region = data1["Region"].value_counts()
Region_i = list(Region.index)
print(Region)
data1.loc[data1["Region"]=="Northern", "State / Union territory (UT)"]
data1.loc[data1["Region"]=="Southern", "State / Union territory (UT)"]

figure = go.Bar(x=Region_i,
               y=Region)
layout = go.Layout(title="Region wise states",
                  barmode='group')
fig = go.Figure(data=figure, layout=layout)
fig.show()

Region = data1["Region"].value_counts()
Region_i = np.array(Region.index)
print(Region_i)

area = []
for i in Region_i:
    area_region = data1.loc[(data1["Region"]==i, ["Area (km2)"])]
    x = np.array(area_region.sum())
    area.append(x)
print(area)
plt.plot(Region_i, area)

plt.show()
    
    
    

pie = go.Pie(labels=data1["State / Union territory (UT)"],
            values=data1["National Share (%)"])
layout = go.Layout(title="% share by each state",
                  height=1000,
                  width=1000)
fig = go.Figure(data=pie, layout=layout)
fig.show()
data2 = pd.read_csv("../input/daily-power-generation-in-india-20172020/file_02.csv")
data2.head()
data2['Date'].value_counts()
#data2.info()
data3 = pd.read_csv("../input/daily-power-generation-in-india-20172020/region_cordinates.csv")
data3.head()
data3.info()
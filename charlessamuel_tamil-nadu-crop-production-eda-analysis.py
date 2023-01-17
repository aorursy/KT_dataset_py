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



import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go
df = pd.read_csv('../input/tamilnadu-cropproduction/Tamilnadu agriculture yield data.csv')

df.sample(10)
df.shape
df.info()
df.describe()
df.isnull().sum()
msno.matrix(df)

plt.show()
df['State_Name'].value_counts()
df.drop('State_Name', axis=1, inplace=True)
df.dropna(how='any', inplace=True)
print("Duplicates:", len(df[df.duplicated()]))
df.District_Name = df.District_Name.apply(lambda x: x.capitalize())
grp = df.groupby("Crop_Year")["Area"].sum().sort_index(ascending=True)
ag_area = pd.DataFrame({'Year': grp.index,

                        'Agricultural Area': grp.values})

ag_area.head()
fig = go.Figure(data=go.Scatter(x = ag_area['Year'], y = ag_area['Agricultural Area'], marker_color = ag_area['Agricultural Area']))

fig.update_layout(title='Agricultural Area over the years',  xaxis = dict(tickmode = 'linear', dtick = 1))

fig.show()
grp_dist = df[df.Crop_Year == 1998].groupby("District_Name")["Area"].sum().sort_values(ascending = False)
dist_df = pd.DataFrame({'District': grp_dist.index, 'Agricultural Area': grp_dist.values})

dist_df.head()
fig = px.bar(dist_df, x='District', y='Agricultural Area', color='Agricultural Area', height=600, width=1000, text='Agricultural Area', title='Agricultural Area in 1998')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
df.Season.value_counts()
df.Crop.value_counts()
se_crop = df.groupby(['Season', 'Crop'])["Production"].sum()
seas_crops = pd.DataFrame({"Production": se_crop}).reset_index()

seas_crops.head()
seas_crops.Season.value_counts()
wy = seas_crops[seas_crops['Season'] == 'Whole Year']

nwy = seas_crops[seas_crops['Season'] != 'Whole Year']

nwy.head()
fig = px.sunburst(nwy, path=['Season', 'Crop'], values='Production')

fig.show()
crop_df = pd.concat([wy.sample(frac=0.4), nwy])
fig = px.sunburst(crop_df, path=['Season', 'Crop'], values='Production')

fig.show()
fig = px.scatter(df, x="Production", y="Area",size="Crop_Year", color="Season", log_x=True, size_max=15, title = "Area and Production in each season")

fig.show()
dist_s = df.groupby(["District_Name", "Season"])["Production"].sum()
kr = pd.DataFrame({"Production": dist_s}).reset_index()

kr = kr.sort_values("Production", ascending=False)

kr = kr[kr.Season != 'Whole Year']

kr.Season.value_counts()
fig = px.bar(kr, "District_Name", y="Production", color="Season", title="Kharif vs Rabi in each District")

fig.show()
fin = df.groupby(["Season","District_Name","Crop"])["Production"].sum()
final_df = pd.DataFrame({"Production": fin}).reset_index()

final_df.sort_values("Production", ascending=False, inplace=True)
fig = go.Figure(data=[go.Table( header=dict(values=list(final_df.columns),

                fill_color='lightblue',

                align='left'),

    cells=dict(values=[final_df.Season, final_df.District_Name, final_df.Crop, final_df.Production],

               fill_color='pink',

               align='left'))

])

fig.show()
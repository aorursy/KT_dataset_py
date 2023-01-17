# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv")

df.info()
df.describe().T
#df[df.Code.isnull()].Entity.unique()
df1 = df.dropna()[df.dropna().Entity!="World"]

cntries = df1.groupby("Code").sum().sort_values(df1.columns[2],ascending=False)
#df.groupby("Year").mean().plot(figsize=(18,10))

px.line(df1.groupby("Year").sum().reset_index(),x="Year",y="Annual CO₂ emissions (tonnes )")

#px.histogram(df[df.Entity=="World"],x="Year",y="Annual CO₂ emissions (tonnes )",nbins=400)
df1.groupby("Year").sum()
#df[df.Entity=="World"]
cntries[cntries.columns[1]][0:10].sort_values().plot(kind="barh")

plt.title("Countries by Most Carbon Emission")
df.dropna()[df.dropna().Code.str.startswith("OWID")].Entity.unique()
#df[df.Entity=="World"][df.columns[3]].plot()
years = df1.groupby("Year").sum().reset_index()
arr = np.array(years[years.columns[1]])

arr = np.insert(arr,0,0)

change = years[years.columns[1]]-arr[0:267]

change.index=years.Year


#change.plot(figsize=(18,10))

#plt.title("Change of Emission By Years")
change

px.line(change,width=2000,title="Change of Carbon Emission by Years")
df1[df1.Year.isin(np.sort(df.Year.unique())[0:50])].groupby("Code").sum()
fig = px.choropleth(df1.sort_values("Year"),locations="Code",

                    hover_name="Code", # column to add to hover information

                    color=df1.columns[3],

                    color_continuous_scale=px.colors.sequential.RdBu,

                    animation_frame="Year",

                    title = "Animated Carbon Emission by Countries"

                   )

fig.show()
df
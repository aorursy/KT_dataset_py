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

!pip install chart_studio
from IPython.display import Image
Image("../input/space2/photo-1541185934-01b600ea069c.jpg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
plt.style.use('ggplot')
url = "../input/all-space-missions-from-1957/Space_Corrected.csv"
data = pd.read_csv(url)

df = data.copy()

df.head()
#Adding a column named "Country"
df["Country"] = df["Location"].str.split(",").str[-1]
df["Country"].value_counts()
#Adding a column named "Year"
df["Year"] = df["Datum"].str.split(",").str[-1]
df["Year"] = df["Year"].str.split(" ").str[1]
df["Year"].value_counts().plot(color='m')
#Replacing bad location

df['Country'] = df['Country'].replace([' Shahrud Missile Test Site'],' Iran')
df['Country'] = df['Country'].replace([' New Mexico'],' USA')
df['Country'] = df['Country'].replace([' Yellow Sea'],' China')
df['Country'] = df['Country'].replace([' Pacific Missile Range Facility'],' USA')
df['Country'] = df['Country'].replace([' Barents Sea'],' Russia')
df['Country'] = df['Country'].replace([' Gran Canaria'],' USA')
df['Country'] = df['Country'].replace([' Pacific Ocean'],' Sea Launch')
df.head()
#Counting Nan Values
nan_val = df[" Rocket"].isna().sum()
nan_val_percentage = int(nan_val*100/df[" Rocket"].shape[0])
print("Percentage of nan values in the Rocket column is :", nan_val_percentage, "%")
import cufflinks as cf
import plotly.express as px
import plotly.offline as py
from plotly.offline import plot
import plotly.graph_objs as go
#Graph : Country by Launches
fig = px.bar(df["Country"].value_counts(ascending=True), orientation="h", color=df["Country"].value_counts(ascending=True), color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=True, labels={'value':'Launches', 
                                'index':'Country',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Country by Launches"
)

fig.show()
group_1 = df.groupby(['Country','Status Mission'])['Country'].count().unstack()
group_1.sort_index(ascending=False)

#Graph : Country by Status Mission
fig = px.bar(group_1, orientation="h", color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=True, labels={'value':'Launches', 
                                'index':'Country',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Country by Status Missions"
)

fig.show()
from IPython.display import display

for c in df["Country"].value_counts().index:
    total = df[df["Country"] == c].count()
    total = total["Country"]
    
    total_failure = total & df[df["Status Mission"] == "Failure"].count()
    total_failure = total_failure["Country"]

    if total_failure > 50/100*total:
        bad_perc = {'Country': [c], 'Failure count': [total-total_failure]}
        
        df_bad_perc = pd.DataFrame(bad_perc)
        
        display(df_bad_perc)
#10 First in a DataFrame
country = df["Country"].value_counts()

df1 = pd.DataFrame(country[:10])
df1.rename(columns={"Country": "Launches"})
sort_company = df["Company Name"].value_counts()
first_sort_company = sort_company[:20]

#Graph : Country by Company Name (First 20)
fig = px.bar(first_sort_company, orientation="h", color=first_sort_company, 
             color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=True, labels={'value':'Launches', 
                                'index':'Country',
                                 'color':'None'
                                })
fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Launches By Company"
)

fig.show()
sort_loc = df["Location"].value_counts()
first_sort_loc = sort_loc[:20]
#12 First in a DataFrame
df2 = pd.DataFrame(first_sort_company[:12])
df2.rename(columns={"Company Name": "Launches"})
#Graph : Country by Location (First 20)
fig = px.bar(first_sort_loc, orientation="h", color=first_sort_loc, color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=True, labels={'value':'Launches', 
                                'index':'Country',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Launches By Location"
)

fig.show()
#Graph : Year by Launches
fig = px.bar(df["Year"].value_counts(), orientation="v", color=df["Year"].value_counts(), color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=True, labels={'value':'Launches', 
                                'index':'Year',
                                 'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Year by Launches"
)

fig.show()
#Status Mission
df3 = pd.DataFrame(df["Status Mission"].value_counts())

fig, ax = plt.subplots(figsize=(10,6))
df3.plot(kind='barh', legend=True, ax=ax, log=True)
ax.set_xlabel('Count')
ax.set_ylabel('Mission Status')
#Status Mission in a DataFrame
df5 = pd.DataFrame(df["Status Mission"].value_counts())
df5.rename(columns={"Status Mission": "Status"})
#Companies by launches for last_decade
last_decade = df[df["Year"] > "2010"]

fig = px.bar(last_decade["Country"].value_counts(), color=last_decade["Country"].value_counts(), orientation="v", color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=False, labels={'value':'Launches', 
                                'index':'Countries',
                                'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Countries by Launches"
)

fig.show()
fig = px.bar(last_decade["Company Name"].value_counts(), color=last_decade["Company Name"].value_counts(), orientation="v", color_continuous_scale=px.colors.sequential.Plasma, 
             log_x=False, labels={'value':'Launches', 
                                'index':'Company',
                                'color':'None'
                                })

fig.update_layout(
    font_color="black",
    title_font_color="red",
    legend_title_font_color="green",
    title_text="Company by Launches"
)

fig.show()
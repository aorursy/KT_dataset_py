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
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
print("All the modules imported!")
dataset_url="https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv"
df=pd.read_csv(dataset_url)
df=df[df["Confirmed"]>0]
df.head()

df.tail()
fig=px.choropleth(df,locations='Country',locationmode='country names',color='Confirmed'
                 ,animation_frame='Date')
fig.update_layout(title_text="Global spread of COVID-19")
fig.show()

fig=px.choropleth(df,locations='Country',locationmode='country names',color='Deaths'
                 ,animation_frame='Date')
fig.update_layout(title_text="Global deaths due to COVID-19")
fig.show()

fig=px.choropleth(df,locations='Country',locationmode='country names',color='Recovered'
                 ,animation_frame='Date')
fig.update_layout(title_text="Patients cured")
fig.show()

df_india=df[df["Country"]=="India"]
df_india.head()
df_india.tail()
df_india["Infection Rate"]=df_india["Confirmed"].diff()
df_india.tail()
df_india=df_india[df_india["Infection Rate"]>0]
df_india.head()
px.line(df_india,y="Confirmed",x="Date")
px.line(df_india,y="Deaths",x="Date")
px.line(df_india,y="Recovered",x="Date")
lockdown_start_date="2020-03-24"
lockdown_end_date="2020-08-17"
fig=px.line(df_india,x="Date",y="Infection Rate")
fig.add_shape(dict(type="line",x0=lockdown_start_date,y0=0,
                  x1=lockdown_start_date,y1=df_india["Infection Rate"].max(),
                  line=dict(color="red",width=2)))
fig.add_annotation(dict(x=lockdown_start_date,y=df_india["Infection Rate"].max(),text='starting date of the lockdown'))



fig.add_shape(dict(type='line',x0=lockdown_end_date,y0=0,
                  x1=lockdown_end_date,y1=df_india["Infection Rate"].max(),
                  line=dict(color="red",width=2)))
fig.add_annotation(dict(x=lockdown_end_date,y=df_india["Infection Rate"].max(),text="lockdown end date"))
fig.show()

print("The maximum rate of infection in India till now is:",df_india["Infection Rate"].max())
df_india["Death Rate"]=df_india["Deaths"].diff()
df_india_death=df_india[df_india["Death Rate"]>0]
df_india_death.head()
fig_death=px.line(df_india_death,x="Date",y="Death Rate")
fig_death.add_shape(dict(type="line",x0=lockdown_start_date,y0=0,
                  x1=lockdown_start_date,y1=df_india["Death Rate"].max(),
                  line=dict(color="red",width=2)))
fig_death.add_annotation(dict(x=lockdown_start_date,y=df_india_death["Death Rate"].max(),text='starting date of the lockdown'))



fig_death.add_shape(dict(type='line',x0=lockdown_end_date,y0=0,
                  x1=lockdown_end_date,y1=df_india_death["Death Rate"].max(),
                  line=dict(color="red",width=2)))
fig_death.add_annotation(dict(x=lockdown_end_date,y=df_india_death["Death Rate"].max(),text="lockdown end date"))
fig_death.show()

df_india.head()
df_india=df_india.dropna()
drop=["Infection Rate","Death Rate"]
#df_india.drop(drop,axis=1,inplace=True)
df_india.groupby("Country").sum().plot.barh()




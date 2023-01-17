# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/country_stats.csv')
df=df.drop(columns=["country_code"])
df.head()
df.dropna(inplace=True)
for i in df.columns:
    print(df[i].value_counts(dropna=False).head())
    print("*******************")
df_pop=pd.DataFrame({"country": df.country_name,"population":df.population})
index_pop=(df_pop["population"].sort_values(ascending=False)).index.values
sorted_pop=df_pop.reindex(index_pop)
sorted_pop.head()
plt.figure(figsize=(15,8))
sns.barplot(x=sorted_pop.country.head(50),y=sorted_pop.population.head(50))
plt.xticks(rotation=90)
plt.xlabel("Countries",size=15)
plt.ylabel("Populations",size=15)
plt.title("Countries Population Bar Plot")
plt.show()
df[df.continent=="Asia"].population.sum()
df[df.continent=="Europe"].population.sum()
df[df.continent=="Americas"].population.sum()
df[df.continent=="Africa"].population.sum()
liste_pop=[df[df.continent=="Asia"].population.sum(),df[df.continent=="Europe"].population.sum(),
           df[df.continent=="Americas"].population.sum(),df[df.continent=="Africa"].population.sum()]
liste_continent=["Asia","Europe","Americas","Africa"]
liste_pop

trace=go.Pie(values=liste_pop, labels= liste_continent,name="Continent",
            hoverinfo="label+percent+name",hole=0.3)
layout=go.Layout(title="Number of Population sorted by Cotinents",
                 annotations= [dict(font=dict(size=20),showarrow=False,text="Populations",x=0.5,y=1)]
                )
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig)
df[df.continent=="Asia"].population_below_poverty_line
df[df.continent=="Europe"].population_below_poverty_line
df[df.continent=="Americas"].population_below_poverty_line
df[df.continent=="Africa"].population_below_poverty_line

trace0=go.Box(y=df[df.continent=="Asia"].population_below_poverty_line, name="poverty of Asia",marker=dict(color="rgba(2,87,222,0.7)"),
             text=df[df.continent=="Asia"].country_name)
trace1=go.Box(y=df[df.continent=="Europe"].population_below_poverty_line, name="poverty of Europe",marker=dict(color="rgba(222,36,176,0.7)"),
             text=df[df.continent=="Europe"].country_name
             )
trace2=go.Box(y=df[df.continent=="Americas"].population_below_poverty_line, name="poverty of America",marker=dict(color="rgba(55,33,7,0.7)"),
             text=df[df.continent=="Americas"].country_name
             )
trace3=go.Box(y=df[df.continent=="Africa"].population_below_poverty_line, name="poverty of Africa",marker=dict(color="rgba(111,3,33,0.7)"),
             text=df[df.continent=="Africa"].country_name
             )
trace4=go.Box(y=df[df.continent=="Oceania"].population_below_poverty_line, name="poverty of Oceania",marker=dict(color="rgba(111,3,33,0.7)"),
             text=df[df.continent=="Oceania"].country_name
             )
data=[trace0,trace1,trace2,trace3,trace4]
iplot(data)
df_pov=pd.DataFrame({"country": df.country_name,"poverty_level":df.population_below_poverty_line,"continents":df.continent})
index_pov=(df_pov["poverty_level"].sort_values(ascending=True)).index.values
sorted_pov=df_pov.reindex(index_pov)



plt.figure(figsize=(15,8))
sns.barplot(x=sorted_pov.country.head(50), y=sorted_pov.poverty_level.head(50))
plt.xticks(rotation=90)
plt.title("Poverty level of top 50 countries")
plt.show()

df[df.continent=="Asia"].life_expectancy.mean()
df[df.continent=="Europe"].life_expectancy.mean()
df[df.continent=="Americas"].life_expectancy.mean()
df[df.continent=="Africa"].life_expectancy.mean()
df[df.continent=="Oceania"].life_expectancy.mean()
df.life_expectancy.mean()
df_cont_name=["Asia","Europe","Americas","Africa","Oceania"]
df_cont_age_mean=[df[df.continent=="Asia"].life_expectancy.mean(),df[df.continent=="Europe"].life_expectancy.mean(),
                 df[df.continent=="Americas"].life_expectancy.mean(),df[df.continent=="Africa"].life_expectancy.mean(),
                 df[df.continent=="Oceania"].life_expectancy.mean()]
df_cont_age_data=pd.DataFrame({"cont_names":df_cont_name, "cont_age":df_cont_age_mean})

trace1=go.Bar(x=df_cont_age_data.cont_names,y=df_cont_age_data.cont_age,
              name="mean age",marker=dict(color="rgba(111,17,35,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)),
              text=df_cont_age_data.cont_names
             )

trace2=go.Bar(x=df_cont_age_data.cont_names,y=[df.life_expectancy.mean(),df.life_expectancy.mean(),df.life_expectancy.mean(),
                                              df.life_expectancy.mean(),df.life_expectancy.mean()],
            name="average age of continents",text="total" 
            )


data=[trace1,trace2]
layout=go.Layout(barmode='group',title="Average age of all continents")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
df50=df.iloc[:50,:]
gni_color=df50.gni
hdi_size=(df50.hdi)*40
trace=dict(x=df50.life_expectancy,y=df50.mean_years_of_schooling,mode="markers",
           marker=dict(color=gni_color,size=hdi_size,showscale=True),text=df50.country_name
          )
layout=dict(xaxis=dict(title="life expectancy"),yaxis=dict(title="years of schooling"),title="Bubble chart")
data=[trace]
fig=dict(data=data,layout=layout)
iplot(fig)
df_matrix=df.loc[:,["hdi","life_expectancy","mean_years_of_schooling"]]
df_matrix["index"]=np.arange(1,len(df_matrix)+1)

fig=ff.create_scatterplotmatrix(df_matrix,diag="box",index="index",colormap="Portland",
                                colormap_type="cat",height=700,width=700
                               )
iplot(fig)
df_log_pop=df.copy()
df_log_pop.population=np.log(df_log_pop.population)
sns.lmplot(x="hdi",y="population",data=df_log_pop)
plt.show()
sns.lmplot(x="hdi",y="life_expectancy",data=df)
plt.show()
df_hdi=pd.DataFrame({"country": df.country_name,"hdi":df.hdi,"continents":df.continent})
index_hdi=(df_hdi["hdi"].sort_values(ascending=False)).index.values
sorted_hdi=df_hdi.reindex(index_hdi)
trace=go.Scatter3d(x=df.gni,y=df.life_expectancy,z=df.expected_years_of_schooling,
                   mode="markers",marker=dict(size=10,color="rgb(111,55,209)"),text=df.country_name
                  )
data=[trace]
layout=go.Layout(margin=dict(l=0,r=0,b=0,t=0))
fig=go.Figure(data=data,layout=layout)
iplot(fig)




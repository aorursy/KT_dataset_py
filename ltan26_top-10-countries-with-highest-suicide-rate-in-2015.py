# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/who_suicide_statistics.csv")
df['suicides_no_fillna']=df.groupby(['country','sex','age']).transform(lambda x: x.fillna(x.mean()))['suicides_no']
df['population_fillna']=df.groupby(['country','sex','age']).transform(lambda x: x.fillna(x.mean()))['population']
df['age']=df['age'].str.replace(' years','')
df.loc[(df['age']=='5-14'),'age']='05-14'
df.head()
df.info()
print("Number of countries:",len(df.country.unique()))
print("Year of oldest record:",df.year.min())
print("Year of newest record:",df.year.max())
df_2015=df[df['year']==2015]
df_2015.head()
temp=df_2015.groupby('country').agg('sum').reset_index()
temp['rate_per_100k_people']=temp['suicides_no_fillna']/(temp['population_fillna']/100000)
temp=temp[temp['population_fillna']!=0].sort_values('rate_per_100k_people',ascending=False).head(10)
top10_rate=temp['country'].tolist()

import math

fig,ax=plt.subplots(nrows=2,ncols=5,sharey=True)
fig.set_figheight(10)
fig.set_figwidth(2*len(top10_rate))
fig.subplots_adjust(hspace=0.4, wspace=0.2)

palette ={"male":"#EE8C1E","female":"#5499C7"}
for i in range(0, (len(top10_rate))):
    temp=df.loc[(df['country']==top10_rate[i]) & (df['year']==2015),['country','sex','age','suicides_no_fillna','population_fillna']]
    temp['rate_per_100k_people']=temp['suicides_no_fillna']/(temp['population_fillna']/100000)
    temp=temp.sort_values('age')
    row=math.floor((i)/5)
    col=(i%5)
    title=str(top10_rate[i]) 
    sns.barplot('age','rate_per_100k_people',hue='sex',data=temp,ax=ax[row][col],palette=palette)
    ax[row][col].set_title(title,fontsize=20)
temp=df_2015.groupby('age').agg('sum').reset_index()
sns.barplot('age','suicides_no_fillna',data=temp,color='#5499C7')
plt.show()
temp=df[df['year']>=2000]
temp=temp.groupby(['country','year']).agg('sum').reset_index()
temp['rate_per_100k_people']=temp['suicides_no_fillna']/(temp['population_fillna']/100000)
temp=temp[temp.population_fillna!=0]
plt.scatter(x='population',y='rate_per_100k_people',data=temp)
plt.show()

import scipy
print("Pearson correlation between population size and suicide rate:",scipy.stats.pearsonr(temp['population'],temp['rate_per_100k_people'])[0])
from bokeh.plotting import figure, show
from bokeh.models import HoverTool
from bokeh.models import ColumnDataSource
from bokeh.io import output_notebook
from bokeh.models import DatetimeTickFormatter

output_notebook()
source_all=[]

temp=df.groupby(['country','year']).agg('sum').reset_index()
temp['rate_per_100k_people']=temp['suicides_no_fillna']/(temp['population_fillna']/100000)
max_count=temp[temp['country'].isin(top10_rate)].rate_per_100k_people.max()

for i in range(0,len(top10_rate)):
    plot_data=temp[temp['country']==top10_rate[i]]
    
for i in range(0,len(top10_rate)):
    plot_data=temp[temp['country']==top10_rate[i]]
    x_vals = plot_data['year'].tolist()
    x = plot_data['year'].tolist()
    y = plot_data['rate_per_100k_people'].tolist()
    desc = plot_data['country'].tolist()

    data = {'x': x,
            'y': y,
            'desc':desc,
            'year':x_vals,
            'count':y
           }
    source = ColumnDataSource(data)
    source_all.append(source)

plot = figure(plot_width=800, plot_height=400,y_range=(0,max_count+0.1*max_count))   
for i in range(0,len(source_all)):
    plot.circle(x='x',y='y', source=source_all[i],fill_color="white",alpha=0.1,name="circle")
    plot.line(x='x',y='y', source=source_all[i],line_width=2,alpha=0.1,hover_line_alpha=1,hover_line_color="#B22222",name="line")
    
hover = HoverTool(show_arrow=False,
                  line_policy='nearest',
                  names=["line"],
                  tooltips=[('desc','@desc'),
                            ('year','@year'),
                            ('suicide rate','@count')]
                     )
plot.add_tools(hover)
show(plot) 

import pycountry
temp=df[df['year']>=2000].copy()
replace_country={'Czech Republic':'Czechia','Hong Kong SAR':'Hong Kong','Iran (Islamic Rep of)':'Iran, Islamic Republic of',
                'Republic of Korea':'Korea, Republic of','Republic of Moldova':'Moldova, Republic of','Reunion':'Réunion',
                'Rodrigues':'Mauritius','Saint Vincent and Grenadines':'Saint Vincent and the Grenadines',
                'TFYR Macedonia':'Macedonia, Republic of','United States of America':'United States',
                'Venezuela (Bolivarian Republic of)':'Venezuela, Bolivarian Republic of',
                'Virgin Islands (USA)':'Virgin Islands, U.S.'}
temp['country'].replace(replace_country,inplace=True)

temp=temp.groupby(['country','year']).agg('sum').reset_index()
temp['rate_per_100k_people']=temp['suicides_no_fillna']/(temp['population_fillna']/100000)
temp=temp[temp['population_fillna']!=0].groupby('country').agg('mean').reset_index()[['country','rate_per_100k_people']]

d={'country':[x.name for x in list(pycountry.countries)],'code':[x.alpha_3 for x in list(pycountry.countries)]}
temp=pd.DataFrame(data=d).merge(temp,on='country',how='left')
temp=temp.fillna(0)
temp.head()
import plotly.plotly as py
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)

data = [ dict(
        type = 'choropleth',
        locations = temp['code'],
        z = temp['rate_per_100k_people'],
        text = temp['country'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            title = 'Suicide Rate'),
      ) ]

layout = dict(
    title = 'Suicide Rate by Country in 21st Century',
    width=1000,
    height=800,
    margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=50,
        pad=0
    ),
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'mercator'
        )
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='choropleth-map')

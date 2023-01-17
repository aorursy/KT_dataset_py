import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.font_manager as fm
from pandas.tools.plotting import parallel_coordinates
import matplotlib.pyplot as plt
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

# matplotlib

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data2015 = pd.read_csv('../input/2015.csv')
data2016 = pd.read_csv('../input/2016.csv')
data2017 = pd.read_csv('../input/2017.csv')
# Columns name was changed
data2015.columns = ['country','region','happiness_rank','happiness_score','standard_error','economy','family','health','freedom','trust','generosity','dystopia']
data2016.columns = ['country','region','happiness_rank','hapiness_score','lower_confidence','upper_confidence','economy','family','health','freedom','trust','generosity','dystopia']
data2017.columns = ['country','happiness_rank','happiness_score','whisker_high','whisker_low','economy','family','health','freedom','generosity','trust','dystopia']
trace = go.Table(
    header = dict(values=list(data2015.columns),
                 fill=dict(color='#C2D4FF'),
                 align=['left']*1),
cells = dict(values=[data2015.country, data2015.region, data2015.happiness_rank, data2015.happiness_score, data2015.standard_error,data2015.economy,data2015.family, data2015.health,data2015.freedom,data2015.trust,data2015.generosity,data2015.dystopia],
                    fill = dict(color = '#F5F8FF'),
                    align = ['left']*1))
data=[trace]
fig = dict(data=data)
iplot(fig)
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(data2015.corr(),annot=True,linewidth =  .5 ,fmt='.1f',ax=ax)
plt.show()
data2015.columns
region_mean = data2015.groupby(['region'])['happiness_score'].mean()
region_mean.sort_values(ascending=False ,inplace=True)
fontsize2use = 12

fig = plt.figure(figsize=(10,5))
plt.xticks(fontsize=fontsize2use)  
plt.yticks(fontsize=fontsize2use)    
fontprop = fm.FontProperties(size=fontsize2use)
ax = fig.add_subplot(111)
ax.set_xlabel('')
ax.set_ylabel('')
plt.bar(region_mean.index,region_mean.values,color='b',alpha=0.3)
plt.xticks(region_mean.index, rotation=90)
plt.title('2015 Hapiness ranking by region')
plt.show()
region_mean = data2015.groupby(['region'])['family'].mean()
region_mean.sort_values(ascending=False ,inplace=True)
fontsize2use = 12

fig = plt.figure(figsize=(10,5))
plt.xticks(fontsize=fontsize2use)  
plt.yticks(fontsize=fontsize2use)    
fontprop = fm.FontProperties(size=fontsize2use)
ax = fig.add_subplot(111)
ax.set_xlabel('')
ax.set_ylabel('')
plt.bar(region_mean.index,region_mean.values,color='b',alpha=0.3)
plt.xticks(region_mean.index, rotation=90)
plt.title('2015 Family ranking by region')
plt.show()
region_mean = data2015.groupby(['region'])['health'].mean()
region_mean.sort_values(ascending=False ,inplace=True)
fontsize2use = 12

fig = plt.figure(figsize=(10,5))
plt.xticks(fontsize=fontsize2use)  
plt.yticks(fontsize=fontsize2use)    
fontprop = fm.FontProperties(size=fontsize2use)
ax = fig.add_subplot(111)
ax.set_xlabel('')
ax.set_ylabel('')
plt.bar(region_mean.index,region_mean.values,color='b',alpha=0.3)
plt.xticks(region_mean.index, rotation=90)
plt.title('Market Share for Each Genre 1995-2017')
plt.show()
region_mean = data2015.groupby(['region'])['trust'].mean()
region_mean.sort_values(ascending=False ,inplace=True)
fontsize2use = 12

fig = plt.figure(figsize=(10,5))
plt.xticks(fontsize=fontsize2use)  
plt.yticks(fontsize=fontsize2use)    
fontprop = fm.FontProperties(size=fontsize2use)
ax = fig.add_subplot(111)
ax.set_xlabel('')
ax.set_ylabel('')
plt.bar(region_mean.index,region_mean.values,color='b',alpha=0.3)
plt.xticks(region_mean.index, rotation=90)
plt.title('2015 Trust ranking by region')
plt.show()
region_mean = data2015.groupby(['region'])['generosity'].mean()
region_mean.sort_values(ascending=False ,inplace=True)
fontsize2use = 12

fig = plt.figure(figsize=(10,5))
plt.xticks(fontsize=fontsize2use)  
plt.yticks(fontsize=fontsize2use)    
fontprop = fm.FontProperties(size=fontsize2use)
ax = fig.add_subplot(111)
ax.set_xlabel('')
ax.set_ylabel('')
plt.bar(region_mean.index,region_mean.values,color='b',alpha=0.3)
plt.xticks(region_mean.index, rotation=90)
plt.title('2015 Generosity ranking by region')
plt.show()
region_mean = data2015.groupby(['region'])['freedom'].mean()
region_mean.sort_values(ascending=False ,inplace=True)
fontsize2use = 12

fig = plt.figure(figsize=(10,5))
plt.xticks(fontsize=fontsize2use)  
plt.yticks(fontsize=fontsize2use)    
fontprop = fm.FontProperties(size=fontsize2use)
ax = fig.add_subplot(111)
ax.set_xlabel('')
ax.set_ylabel('')
plt.bar(region_mean.index,region_mean.values,color='b',alpha=0.3)
plt.xticks(region_mean.index, rotation=90)
plt.title('2015 Freedom ranking by region')
plt.show()
sorted_happiness = data2015.sort_values('happiness_score', ascending=False)
fontsize2use = 12

fig = plt.figure(figsize=(10,5))
plt.xticks(fontsize=fontsize2use)  
plt.yticks(fontsize=fontsize2use)    
fontprop = fm.FontProperties(size=fontsize2use)
ax = fig.add_subplot(111)
ax.set_xlabel('')
ax.set_ylabel('')
plt.bar(sorted_happiness.country.head(10),sorted_happiness.happiness_score.head(10),color='b',alpha=0.3)
plt.xticks(sorted_happiness.country.head(10), rotation=90)
plt.title('2015 Happiness ranking by country')
plt.show()
data = dict(type = 'choropleth', 
           locations = data2015['country'],
           locationmode = 'country names',
           z = data2015['happiness_rank'], 
           text = data2015['country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = '2015 Global Happiness', 
                 geo = dict(showframe = False, 
                       projection = {'type': 'equirectangular'}))
fig = go.Figure(data = [data], layout=layout)
iplot(fig)
#Horizontal bar plot
region_lists=list(data2015['country'].head(15))
share_economy=[]
share_family=[]
share_health=[]
share_freedom=[]
share_trust=[]
for each in region_lists:
    region=data2015[data2015['country']==each]
    share_economy.append(sum(region.economy)/len(region))
    share_family.append(sum(region.family)/len(region))
    share_health.append(sum(region.health)/len(region))
    share_freedom.append(sum(region.freedom)/len(region))
    share_trust.append(sum(region.trust)/len(region))

#Visualization
f,ax = plt.subplots(figsize = (15,13))
sns.set_color_codes("pastel")
sns.barplot(x=share_economy,y=region_lists,color='g',label="Economy")
sns.barplot(x=share_family,y=region_lists,color='b',label="Family")
sns.barplot(x=share_health,y=region_lists,color='c',label="Health")
sns.barplot(x=share_freedom,y=region_lists,color='y',label="Freedom")
sns.barplot(x=share_trust,y=region_lists,color='r',label="Trust")
ax.legend(loc="lower right",frameon = True)
ax.set(xlabel='Percentage', ylabel='Country',title = "2015 Top 15 Country")
plt.show()
happiness2015_rank = data2015.sort_values('happiness_score', ascending=False)
happiness2016_rank = data2016.sort_values('hapiness_score', ascending=False)
happiness2017_rank = data2017.sort_values('happiness_score', ascending=False)

trace1 = go.Bar(
            x=happiness2015_rank.country.head(15),
            y=happiness2015_rank.happiness_score.head(15),
            opacity=0.65,
            name='2015 Happiness'
        )

trace2 = go.Bar(
            x=happiness2016_rank.country.head(15),
            y=happiness2016_rank.hapiness_score.head(15),
            opacity=0.65,
            name='2016 Happiness'
        )
trace3 = go.Bar(
            x=happiness2017_rank.country.head(15),
            y=happiness2017_rank.happiness_score.head(15),
            opacity=0.65,
            name='2017 Happiness'
        )

fig = tools.make_subplots(rows=3, cols=1, subplot_titles=())
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)

fig['layout'].update(height=650, width=900, title='Ranking of Happiness by Year')
iplot(fig)
freedom2015_rank = data2015.sort_values('freedom', ascending=False)
freedom2016_rank = data2016.sort_values('freedom', ascending=False)
freedom2017_rank = data2017.sort_values('freedom', ascending=False)

trace1 = go.Bar(
            x=freedom2015_rank.country.head(15),
            y=freedom2015_rank.freedom.head(15),
            opacity=0.65,
            name='2015 Freedom'
        )

trace2 = go.Bar(
            x=freedom2016_rank.country.head(15),
            y=freedom2016_rank.freedom.head(15),
            opacity=0.65,
            name='2016 Freedom'
        )
trace3 = go.Bar(
            x=freedom2017_rank.country.head(15),
            y=freedom2017_rank.freedom.head(15),
            opacity=0.65,
            name='2017 Freedom'
        )

fig = tools.make_subplots(rows=3, cols=1, subplot_titles=())
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)

fig['layout'].update(height=650, width=900, title='Ranking of Freedom by Year')
iplot(fig)
generosity2015_rank = data2015.sort_values('generosity', ascending=False)
generosity2016_rank = data2016.sort_values('generosity', ascending=False)
generosity2017_rank = data2017.sort_values('generosity', ascending=False)

trace1 = go.Bar(
            x=generosity2015_rank.country.head(15),
            y=generosity2015_rank.generosity.head(15),
            opacity=0.65,
            name='2015 Generosity'
        )

trace2 = go.Bar(
            x=generosity2016_rank.country.head(15),
            y=generosity2016_rank.generosity.head(15),
            opacity=0.65,
            name='2016 Generosity'
        )
trace3 = go.Bar(
            x=generosity2017_rank.country.head(15),
            y=generosity2017_rank.generosity.head(15),
            opacity=0.65,
            name='2017 Generosity'
        )

fig = tools.make_subplots(rows=3, cols=1, subplot_titles=())
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)

fig['layout'].update(height=700, width=900, title='Ranking of Generosity by Year')
iplot(fig)
trace1 = go.Bar(
            x= data2015.sort_values('economy', ascending=False).country.head(15),
            y=data2015.sort_values('economy', ascending=False).economy.head(15),
            opacity=0.65,
            name='2015 Economy'
        )

trace2 = go.Bar(
            x= data2016.sort_values('economy', ascending=False).country.head(15),
            y= data2016.sort_values('economy', ascending=False).economy.head(15),
            opacity=0.65,
            name='2016 Economy'
        )
trace3 = go.Bar(
            x=data2017.sort_values('economy', ascending=False).country.head(15),
            y=data2017.sort_values('economy', ascending=False).economy.head(15),
            opacity=0.65,
            name='2017 Economy'
        )


fig = tools.make_subplots(rows=3, cols=1, subplot_titles=())
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)

fig['layout'].update(height=550, width=900, title='2015 vs 2016 vs 2017')
iplot(fig)
trace1 = go.Bar(
            x= data2015.sort_values('family', ascending=False).country.head(15),
            y=data2015.sort_values('family', ascending=False).family.head(15),
            opacity=0.65,
            name='2015 Family'
        )

trace2 = go.Bar(
            x= data2016.sort_values('family', ascending=False).country.head(15),
            y= data2016.sort_values('family', ascending=False).family.head(15),
            opacity=0.65,
            name='2016 Family'
        )
trace3 = go.Bar(
            x=data2017.sort_values('family', ascending=False).country.head(15),
            y=data2017.sort_values('family', ascending=False).family.head(15),
            opacity=0.65,
            name='2017 Family'
        )


fig = tools.make_subplots(rows=3, cols=1, subplot_titles=())
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)

fig['layout'].update(height=550, width=900, title='2015 vs 2016 vs 2017')
iplot(fig)
trace1 = go.Bar(
            x= data2015.sort_values('freedom', ascending=False).country.head(15),
            y=data2015.sort_values('freedom', ascending=False).freedom.head(15),
            opacity=0.65,
            name='2015 Freedom'
        )

trace2 = go.Bar(
            x= data2016.sort_values('freedom', ascending=False).country.head(15),
            y= data2016.sort_values('freedom', ascending=False).freedom.head(15),
            opacity=0.65,
            name='2016 Freedom'
        )
trace3 = go.Bar(
            x=data2017.sort_values('freedom', ascending=False).country.head(15),
            y=data2017.sort_values('freedom', ascending=False).freedom.head(15),
            opacity=0.65,
            name='2017 freedom'
        )


fig = tools.make_subplots(rows=3, cols=1, subplot_titles=())
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)

fig['layout'].update(height=600, width=900, title='2015 vs 2016 vs 2017')
iplot(fig)
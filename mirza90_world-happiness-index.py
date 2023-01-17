# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode(connected=True)
import matplotlib
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data15=pd.read_csv('../input/2015.csv')
data16=pd.read_csv('../input/2016.csv')
data17=pd.read_csv('../input/2017.csv')
data17.head()
fig = plt.figure();
fig.suptitle('Happiness Score by Year', fontsize=14, fontweight='bold');

ax = fig.add_subplot(111);
ax.boxplot([data15['Happiness Score'],data16['Happiness Score'],data17['Happiness.Score']],labels=['2015','2016','2017']);

ax.set_xlabel('Year');
ax.set_ylabel('Score');

plt.show();

scores=pd.DataFrame(data={'2015':data15['Happiness Score'],'2016':data16['Happiness Score'],'2017':data17['Happiness.Score']})

sns.boxplot(data=scores,palette='Set3');
europe=['Switzerland','Iceland','Denmark','Norway','Finland','Netherlands','Sweden','Austria','Luxembourg','Ireland','Belgium','United Kingdom','Germany','France','Czech Republic','Spain','Malta','Slovakia','Italy','Moldova','Slovenia','Lithuania','Belarus','Poland','Croatia','Russia','North Cyprus','Cyprus','Kosovo','Turkey','Montenegro','Romania','Serbia','Portugal','Latvia','Macedonia','Albania','Bosnia and Herzegovina','Greece','Hungary','Ukraine','Bulgaria']
data15['InEurope']=(data15['Country'].isin(europe))
data16['InEurope']=(data16['Country'].isin(europe))
data17['InEurope']=(data17['Country'].isin(europe))

data15['Year']=2015
data16['Year']=2016
data17['Year']=2017
data17.rename(columns = {'Happiness.Score':'Happiness Score'}, inplace = True)
sns.boxplot(x='Year',y='Happiness Score',hue='InEurope',data=pd.concat([data15[['Year','Happiness Score','InEurope']],data16[['Year','Happiness Score','InEurope']],data17[['Year','Happiness Score','InEurope']]]),palette='Set2');
sns.violinplot(x='Year',y='Happiness Score',hue='InEurope',data=pd.concat([data15[['Year','Happiness Score','InEurope']],data16[['Year','Happiness Score','InEurope']],data17[['Year','Happiness Score','InEurope']]]),palette='Set2');
xdata=['2015','2016','2017']
ydata=[data15['Happiness Score'],data16['Happiness Score'],data17['Happiness Score']]

colors=['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)']
traces=[]
for xd, yd, color in zip(xdata, ydata, colors):
        traces.append(go.Box(
            y=yd,
            name=xd,

            marker=dict(
                color=color,
            )
        ))
        

fig = go.Figure(data=traces)
iplot(fig)
corr = data15.iloc[:,:-1].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=sns.color_palette("RdBu_r", 15), vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
changedata=pd.DataFrame([])
changedata['Country']=data17['Country']
changedata['Change']=(data17['Happiness Score']-data15['Happiness Score'])/data15['Happiness Score']
changedata.dropna(axis=1)

cd_filtered=changedata[np.abs(changedata.Change)>0.01] #let's focus on the countries with at least 1% change

cd_filtered=cd_filtered.sort_values('Change',ascending=False);
sns.set(font_scale=1) 
fig, axes = plt.subplots(1,1,figsize=(10, 20))

colorspal = sns.color_palette('husl', len(cd_filtered['Country']))
sns.barplot(cd_filtered.Change,cd_filtered.Country, palette = colorspal);
axes.set(title='Percentual change in Happiness Score from 2015-2017');
bars = [go.Bar(
            x=cd_filtered.Change,
            y=cd_filtered.Country,
            text=cd_filtered.Country,
            orientation = 'h'
)]

layout=dict(
    title='Percentual change in Happiness Score from 2015-2017',
    yaxis=dict(
        showticklabels=False
    )
)
fig = go.Figure(data=bars, layout=layout)

iplot(fig, filename='horizontal-bar')
data17.rename(columns = {'Economy..GDP.per.Capita.':'GDP per Capita'}, inplace = True)
plt.scatter(data17['GDP per Capita'],data17['Happiness Score'])
plt.xlabel('GDP per Capita');
plt.ylabel('Happiness Score');
sns.jointplot(data17['GDP per Capita'],data17['Happiness Score'],color=matplotlib.colors.hex2color('#663399'));
trace = go.Scatter(
    x = data17['GDP per Capita'],
    y = data17['Happiness Score'],
    text=data17['Country'],
    mode = 'markers'
)
layout = dict(title = '',
              yaxis = dict(zeroline = False,
                          title='Happiness Score'),
              xaxis = dict(zeroline = False,
                          title='GDP per Capita'),
              autosize=False,
    width=500,
    height=500,
             )
fig=dict(data=[trace],layout=layout)
iplot(fig, filename='basic-scatter')
data17.replace('United Kingdom','United Kingdom of Great Britain and Northern Ireland',inplace=True)
data17.replace('Czech Republic','Czechia',inplace=True)
data17.replace('Taiwan Province of China','Taiwan, Province of China',inplace=True)
data17.replace('Russia','Russian Federation',inplace=True)
data17.replace('South Korea','Korea, Republic of',inplace=True)
data17.replace('Moldova','Moldova, Republic of',inplace=True)
data17.replace('Bolivia','Bolivia, Plurinational State of',inplace=True)
data17=data17[data17['Country']!='North Cyprus']
data17.replace('Hong Kong S.A.R., China','Hong Kong', inplace=True)
data17=data17[data17['Country']!='Kosovo']
data17.replace('Venezuela','Venezuela, Bolivarian Republic of',inplace=True)
data17.replace('Macedonia','Macedonia, the former Yugoslav Republic of',inplace=True)
data17.replace('Vietnam','Viet Nam',inplace=True)
data17.replace('Palestinian Territories','Palestine, State of',inplace=True)
data17.replace('Iran','Iran, Islamic Republic of',inplace=True)
data17.replace('Congo (Brazzaville)','Congo',inplace=True)
data17.replace('Congo (Kinshasa)','Congo, Democratic Republic of the',inplace=True)
data17.replace('Ivory Coast',"Côte d'Ivoire",inplace=True)
data17.replace('Syria','Syrian Arab Republic',inplace=True)
data17.replace('Tanzania','Tanzania, United Republic of',inplace=True)
import iso3166
def getalpha3(row):
    return iso3166.countries.get(row['Country']).alpha3

data17['Code']=data17.apply(lambda row: getalpha3(row),axis=1)
data = [ dict(
        type = 'choropleth',
        locations = data17['Code'],
        z = data17['Happiness Score'],
        text = data17['Country'],
        colorscale = 'Jet',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = ''),
      ) ]

layout = dict(
    title = 'World Happiness Index',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )
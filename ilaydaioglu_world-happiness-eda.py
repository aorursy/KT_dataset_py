import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import random
import seaborn as sbr
d15 = pd.read_csv('../input/world-happiness/2015.csv')
d16 = pd.read_csv('../input/world-happiness/2016.csv')
d17 = pd.read_csv('../input/world-happiness/2017.csv')
d18 = pd.read_csv('../input/world-happiness/2018.csv')
d19 = pd.read_csv('../input/world-happiness/2019.csv')
print(d15.info())
print('-' * 100)
print(d15.describe())
print(d16.info())
print('-' * 100)
print(d16.describe())
print(d17.info())
print('-' * 100)
print(d17.describe())
print(d18.info())
print('-' * 100)
print(d18.describe())
print(d19.info())
print('-' * 100)
print(d19.describe())
d15.drop(columns="Standard Error",inplace=True,errors="ignore")
d15.drop(columns="Dystopia Residual",inplace=True,errors="ignore")
d15.drop(columns="Region",inplace=True,errors="ignore")
d16.drop(columns="Lower Confidence Interval",inplace=True,errors="ignore")
d16.drop(columns="Upper Confidence Interval",inplace=True,errors="ignore")
d16.drop(columns="Dystopia Residual",inplace=True,errors="ignore")
d16.drop(columns="Region",inplace=True,errors="ignore")
del d17["Whisker.high"]
del d17["Whisker.low"]
d17.drop(columns="Dystopia.Residual",inplace=True,errors="ignore")
d15=d15.rename(columns={"Happiness Rank":"Overall rank",
                            "Country" : "Country or region",
                            "Happiness Score":"Score",
                            "Economy (GDP per Capita)":"GDP per capita",
                            "Family" : "Social support",
                            "Health (Life Expectancy)":"Healthy life expectancy",
                            "Trust (Government Corruption)":"Perceptions of corruption"})
d16=d16.rename(columns={"Country" : "Country or region",
                           "Happiness Rank":"Overall rank",
                            "Happiness Score":"Score",
                            "Economy (GDP per Capita)":"GDP per capita",
                            "Family" : "Social support",
                            "Health (Life Expectancy)":"Healthy life expectancy",
                            "Trust (Government Corruption)":"Perceptions of corruption"})
d17=d17.rename(columns={"Country" : "Country or region",
                           "Happiness.Rank":"Overall rank",
                            "Happiness.Score":"Score",
                            "Economy..GDP.per.Capita.":"GDP per capita",
                            "Family" : "Social support",
                            "Health..Life.Expectancy.":"Healthy life expectancy",
                            "Trust..Government.Corruption.":"Perceptions of corruption"})
d18=d18.rename(columns={"Freedom to make life choices" : "Freedom"})
d19=d19.rename(columns={"Freedom to make life choices" : "Freedom"})
d15 = d15.reindex(columns= d16.columns)
#we arranged it according to d16
d17 = d17.reindex(columns= d16.columns)
d18 = d18.reindex(columns= d16.columns)
d19 = d19.reindex(columns= d16.columns)
d15.head()
d16.head()
d17.head()
d18.head()
d19.head()
d18['Year']='2018'
d19['Year']='2019'
d15['Year']='2015'
d16['Year']='2016'
d17['Year']='2017'

data1=d15.filter(['Country or region','GDP per capita',"Year"],axis=1)
data2=d16.filter(['Country or region','GDP per capita',"Year"],axis=1)
data3=d17.filter(['Country or region','GDP per capita','Year'],axis=1)
data4=d18.filter(['Country or region','GDP per capita',"Year"],axis=1)
data5=d19.filter(['Country or region','GDP per capita','Year'],axis=1)
data1=data1.append([data2,data3,data4,data5])

plt.figure(figsize=(10,8))
d = data1[data1['Country or region']=='India']
sns.lineplot(x="Year", y="GDP per capita",data=d,label='Venezuela')
d = data1[data1['Country or region']=='United States']
sns.lineplot(x="Year", y="GDP per capita",data=d,label='Norway')
d = data1[data1['Country or region']=='Finland']
sns.lineplot(x="Year", y="GDP per capita",data=d,label='Finland')
d = data1[data1['Country or region']=='United Kingdom']
sns.lineplot(x="Year", y="GDP per capita",data=d,label="Colombia")
d = data1[data1['Country or region']=='Canada']
sns.lineplot(x="Year", y="GDP per capita",data=d,label='Turkey')


plt.title("GDP per capita 2015-2019")
init_notebook_mode(connected=True)

trace1 = go.Scatter3d(
    x=d19["GDP per capita"],
    y=d19["Healthy life expectancy"],
    z=d19["Social support"],
    mode='markers',
    name = "2019",
    marker=dict(
        color='rgb(240,128,128)',
        size=12,
        line=dict(
            color='rgb(255, 255, 255)',
            width=0.1
        )))
trace2 = go.Scatter3d(
    x=d18["GDP per capita"],
    y=d18["Healthy life expectancy"],
    z=d18["Social support"],
    mode='markers',
    name = "2018",
    marker=dict(
        color='rgb(0 ,128, 128)',
        size=12,
        line=dict(
            color='rgb(255, 255, 255)',
            width=0.1
        )))
trace3 = go.Scatter3d(
    x=d17["GDP per capita"],
    y=d17["Healthy life expectancy"],
    z=d17["Social support"],
    mode='markers',
    name = "2017",
    marker=dict(
        color='rgb(202 ,255, 112)',
        size=12,
        line=dict(
            color='rgb(255, 255, 255)',
            width=0.1
        )))
trace4 = go.Scatter3d(
    x=d16["GDP per capita"],
    y=d16["Healthy life expectancy"],
    z=d16["Social support"],
    mode='markers',
    name = "2016",
    marker=dict(
        color='rgb(131,111 ,255)',
        size=12,
        line=dict(
            color='rgb(255, 255, 255)',
            width=0.1
        )))

trace5 = go.Scatter3d(
    x=d15["GDP per capita"],
    y=d15["Healthy life expectancy"],
    z=d15["Social support"],
    mode='markers',
    name = "2015",
    marker=dict(
        color='rgb(126, 192 ,238)',
        size=12,
        line=dict(
            color='rgb(255, 255, 255)',
            width=0.1
        )))


data = [trace1, trace2, trace3,trace4,trace5]
layout = go.Layout(
    title = '2015,2016,2017,2018 and 2019 values',
    scene = dict(xaxis = dict(title='GDP per capita'),
                yaxis = dict(title='Healthy life expectancy'),
                zaxis = dict(title=' Social support'),),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


fig = go.Figure()
fig.add_trace(go.Box(x=d19.Score, name='2019'))
fig.add_trace(go.Box(x=d18.Score, name='2018'))
fig.add_trace(go.Box(x=d17.Score, name='2017'))
fig.add_trace(go.Box(x=d16.Score, name='2016'))
fig.add_trace(go.Box(x=d15.Score, name='2015'))
fig.show()
d_all = pd.concat([d15,d16,d17,d18,d19])
d_all
d_all.info()
d_all.loc[d_all['Country or region']=='Canada']
d_all.loc[d_all['Country or region']=='Kenya']
d_all.loc[d_all['Country or region']=='Suriname']
d_all.loc[d_all['Country or region']=='Swaziland']
d15.drop(d15.index[147],inplace=True)
d15.drop(d15.index[139],inplace=True)
d15.drop(d15.index[136],inplace=True)
d15.drop(d15.index[125],inplace=True)
d15.drop(d15.index[117],inplace=True)
d15.drop(d15.index[100],inplace=True)
d15.drop(d15.index[98],inplace=True)
d15.drop(d15.index[96],inplace=True)
d15.drop(d15.index[93],inplace=True)
d15.drop(d15.index[92],inplace=True)
d15.drop(d15.index[90],inplace=True)
d15.drop(d15.index[39],inplace=True)
d15.drop(d15.index[21],inplace=True)
d16.drop(d16.index[142],inplace=True)
d16.drop(d16.index[140],inplace=True)
d16.drop(d16.index[137],inplace=True)
d16.drop(d16.index[132],inplace=True)
d16.drop(d16.index[112],inplace=True)
d16.drop(d16.index[101],inplace=True)
d16.drop(d16.index[96],inplace=True)
d16.drop(d16.index[94],inplace=True)
d16.drop(d16.index[75],inplace=True)
d16.drop(d16.index[51],inplace=True)
d16.drop(d16.index[39],inplace=True)
d16.drop(d16.index[14],inplace=True)

d17.drop(d17.index[154],inplace=True)
d17.drop(d17.index[146],inplace=True)
d17.drop(d17.index[139],inplace=True)
d17.drop(d17.index[138],inplace=True)
d17.drop(d17.index[129],inplace=True)
d17.drop(d17.index[112],inplace=True)
d17.drop(d17.index[110],inplace=True)
d17.drop(d17.index[92],inplace=True)
d17.drop(d17.index[91],inplace=True)
d17.drop(d17.index[49],inplace=True)

d18.drop(d18.index[154],inplace=True)
d18.drop(d18.index[153],inplace=True)
d18.drop(d18.index[141],inplace=True)
d18.drop(d18.index[140],inplace=True)
d18.drop(d18.index[136],inplace=True)
d18.drop(d18.index[122],inplace=True)
d18.drop(d18.index[118],inplace=True)
d18.drop(d18.index[109],inplace=True)
d18.drop(d18.index[97],inplace=True)
d18.drop(d18.index[88],inplace=True)
d18.drop(d18.index[48],inplace=True)
d19.drop(d19.index[155],inplace=True)
d19.drop(d19.index[154],inplace=True)
d19.drop(d19.index[143],inplace=True)
d19.drop(d19.index[141],inplace=True)
d19.drop(d19.index[134],inplace=True)
d19.drop(d19.index[122],inplace=True)
d19.drop(d19.index[119],inplace=True)
d19.drop(d19.index[112],inplace=True)
d19.drop(d19.index[111],inplace=True)
d19.drop(d19.index[104],inplace=True)
d19.drop(d19.index[83],inplace=True)
d2015=d15.sort_values(by="Country or region").copy()
d2016=d16.sort_values(by="Country or region").copy()
d2017=d17.sort_values(by="Country or region").copy()
d2018=d18.sort_values(by="Country or region").copy()
d2019=d19.sort_values(by="Country or region").copy()
d2015.index=range(len(d2015))
d2016.index=range(len(d2016))
d2017.index=range(len(d2017))
d2018.index=range(len(d2018))
d2019.index=range(len(d2019))
d_total = pd.DataFrame()
d_total["Country or region"] = d2019["Country or region"]
d_total["Overall rank"] = d2019["Overall rank"]
d_total["Score"] = (d2019["Score"]+d2018["Score"]+d2017["Score"]+d2016["Score"]+d2015["Score"])/5
d_total["GDP per capita"] = (d2019["GDP per capita"]+d2018["GDP per capita"]+d2017["GDP per capita"]+d2016["GDP per capita"]+d2015["GDP per capita"])/5
d_total["Social support"] = (d2019["Social support"]+d2018["Social support"]+d2017["Social support"]+d2016["Social support"]+d2015["Social support"])/5
d_total["Healthy life expectancy"] = (d2019["Healthy life expectancy"]+d2018["Healthy life expectancy"]+d2017["Healthy life expectancy"]+d2016["Healthy life expectancy"]+d2015["Healthy life expectancy"])/5
d_total["Freedom"] =  (d2019["Freedom"]+d2018["Freedom"]+d2017["Freedom"]+d2016["Freedom"]+d2015["Freedom"])/5
d_total["Perceptions of corruption"] =  (d2019["Perceptions of corruption"]+d2018["Perceptions of corruption"]+d2017["Perceptions of corruption"]+d2016["Perceptions of corruption"]+d2015["Perceptions of corruption"])/5
d_total["Generosity"] =  (d2019["Generosity"]+d2018["Generosity"]+d2017["Generosity"]+d2016["Generosity"]+d2015["Generosity"])/5

d_total = d_total.sort_values(['Score'],ascending = False)
d_total["Overall rank"] = d_total.reset_index().index
d_total.drop(["Overall rank"],axis = 1)
d_total.info()
d_total.describe()
generated_color_list = []

for i in range(len(d_total.columns) - 2):

    generated_color_list.append("%06x" % random.randint(0, 0xFFFFFF))

d_total.style \
.bar(subset = d_total.columns[2], color= '#'+ generated_color_list[0]) \
.bar(subset = d_total.columns[3], color= '#'+ generated_color_list[1]) \
.bar(subset = d_total.columns[4], color= '#'+ generated_color_list[2]) \
.bar(subset = d_total.columns[5], color= '#'+ generated_color_list[3]) \
.bar(subset = d_total.columns[6], color= '#'+ generated_color_list[4]) \
.bar(subset = d_total.columns[7], color= '#'+ generated_color_list[5]) \
.bar(subset = d_total.columns[8], color= '#'+ generated_color_list[6])


d_total.head()
original=d_total.copy()
def highlight_max(s):    
    is_max = s == s.max()
    return ['background-color: limegreen' if v else '' for v in is_max]
 
d_total.style.apply(highlight_max, subset=['Score','GDP per capita','Social support','Healthy life expectancy','Freedom','Generosity','Perceptions of corruption'])
data = dict(type = 'choropleth', 
           locations= d_total['Country or region'],
           locationmode = 'country names',
           colorscale ='RdYlGn',
           z = d_total['Score'], 
           text = d_total['Country or region'],
           colorbar = {'title':'Happiness Score'})

layout = dict(title = 'Geographical Visualization of Happiness Score', 
              geo = dict(showframe = True, projection = {'type': 'azimuthal equal area'}))

choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
f,ax=plt.subplots(figsize=(10,10))
sbr.heatmap(d_total.corr(),annot=True,fmt=".1f",linewidths=.2,cmap="Spectral",ax=ax,linecolor="black")
plt.show()
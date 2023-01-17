import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  
import warnings            
warnings.filterwarnings("ignore") 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
d_2015=pd.read_csv('/kaggle/input/world-happiness/2015.csv')
d_2016=pd.read_csv('/kaggle/input/world-happiness/2016.csv')
d_2017=pd.read_csv('/kaggle/input/world-happiness/2017.csv')
d_2018=pd.read_csv('/kaggle/input/world-happiness/2018.csv')
d_2019=pd.read_csv('/kaggle/input/world-happiness/2019.csv')
d_=[d_2015,d_2016,d_2017,d_2018,d_2019]
display(d_2015.head())
display(d_2016.head())
display(d_2017.head())
display(d_2018.head())
display(d_2019.head())
display(list(d_2015.columns))
display(list(d_2016.columns))
display(list(d_2017.columns))
display(list(d_2018.columns))
display(list(d_2019.columns))

#d_2019.head()
d_2015.rename(columns={"Happiness Rank": "Rank","Happiness Score":"Score","Economy (GDP per Capita)":"GDP","Health (Life Expectancy)":"Healthy","Trust (Government Corruption)":"Trust"},inplace=True)  
d_2016.rename(columns={"Happiness Rank": "Rank","Happiness Score":"Score","Economy (GDP per Capita)":"GDP","Health (Life Expectancy)":"Healthy","Trust (Government Corruption)":"Trust"},inplace=True)  
d_2017.rename(columns={"Happiness.Rank": "Rank","Happiness.Score":"Score","Economy..GDP.per.Capita.":"GDP","Health..Life.Expectancy.":"Healthy","Trust..Government.Corruption.":"Trust"},inplace=True)  
d_2018.rename(columns={"Country or region": "Country", "Overall rank": "Rank","Score":"Score","GDP per capita":"GDP","Healthy life expectancy":"Healthy","Freedom to make life choices":"Freedom","Perceptions of corruption":"Trust"},inplace=True) 
d_2019.rename(columns={"Country or region": "Country", "Overall rank": "Rank","GDP per capita":"GDP","Healthy life expectancy":"Healthy","Freedom to make life choices":"Freedom","Perceptions of corruption":"Trust"},inplace=True)  
d_2019.head()
display(d_2015.info())
display(d_2016.info())
display(d_2017.info())
display(d_2018.info())
display(d_2019.info())
d_2015_C=list(d_2015["Country"])
d_2016_C=list(d_2016["Country"])
d_2017_C=list(d_2017["Country"])
d_2018_C=list(d_2018["Country"])
d_2019_C=list(d_2019["Country"])

def Equal(x,y,z,t,m):
    Eq_list=[]
    for i in x:
        for j in y:
            if i==j:
                for k in z:
                    if i==k:
                        for l in t:
                            if i==l:
                                for n in m:
                                    if i==n:
                                        Eq_list.append(i)                                        
    
    return Eq_list

useful_country=Equal(d_2015_C,d_2016_C,d_2017_C,d_2018_C,d_2019_C)
print(len(useful_country))
# print(d_2015_C)
# print(useful_country)

def Different(x):
    dif_list=[]
    for i in x:
        if i not in useful_country:
            dif_list.append(i)
    return dif_list        
 

d_list=Different(d_2015_C)
for i in d_list:
    d_2015.drop(d_2015.loc[d_2015['Country']==i].index, inplace=True)
    
d_list=Different(d_2016_C)
for i in d_list:
    d_2016.drop(d_2016.loc[d_2016['Country']==i].index, inplace=True)

d_list=Different(d_2017_C)
for i in d_list:
    d_2017.drop(d_2017.loc[d_2017['Country']==i].index, inplace=True)

d_list=Different(d_2018_C)
for i in d_list:
    d_2018.drop(d_2018.loc[d_2018['Country']==i].index, inplace=True)
   
d_list=Different(d_2019_C)
for i in d_list:
    d_2019.drop(d_2019.loc[d_2019['Country']==i].index, inplace=True)
        
# Dif_Drop(d_2015_C) 
# Different(d_2015_C)
# d_2015_C.drop(d_2015_C.loc[d_2015_C['Country']=='Oman'].index, inplace=True)
# d_2015_C.drop(d_2015_C.index[d_2015_C['Country']=='Oman'], inplace = True)
    
# a= Different(d_2019_C)
# print(len(a))
# print(Different(d_2015_C))  
# print(Different(d_2016_C))  
# print(Different(d_2017_C)) 
# print(Different(d_2018_C))
# print(Different(d_2019_C))  

#indexleri resetliyoruz

d_2015 = d_2015.reset_index()
del d_2015['index']
d_2016 = d_2016.reset_index()
del d_2016['index']
d_2017 = d_2017.reset_index()
del d_2017['index']
d_2018 = d_2018.reset_index()
del d_2018['index']
d_2019 = d_2019.reset_index()
del d_2019['index']



display(d_2015.info())
display(d_2016.info())
display(d_2017.info())
display(d_2018.info())
display(d_2019.info())
    
display(d_2015.head())
display(d_2016.head())
display(d_2017.head())
display(d_2018.head())
display(d_2019.head())
region_list=list(d_2015['Region'].unique())
region_list
Western_Europe=list(d_2015[d_2015["Region"]=="Western Europe"]["Country"])
North_America=list(d_2015[d_2015["Region"]=="North America"]["Country"])
Australia_New_Zealand=list(d_2015[d_2015["Region"]=="Australia and New Zealand"]["Country"])
Middle_East_Northern_Africa=list(d_2015[d_2015["Region"]=="Middle East and Northern Africa"]["Country"])
Latin_America_Caribbean=list(d_2015[d_2015["Region"]=="Latin America and Caribbean"]["Country"])
Southeastern_Asia=list(d_2015[d_2015["Region"]=="Southeastern Asia"]["Country"])
Central_Eastern_Europe=list(d_2015[d_2015["Region"]=="Central and Eastern Europe"]["Country"])
Eastern_Asia=list(d_2015[d_2015["Region"]=="Eastern Asia"]["Country"])
Sub_Saharan_Africa=list(d_2015[d_2015["Region"]=="Sub-Saharan Africa"]["Country"])
Southern_Asia=list(d_2015[d_2015["Region"]=="Southern Asia"]["Country"])

def FixRegion(country_):
    if country_ in Western_Europe:
        return "Western_Europe"
    elif country_ in North_America:
        return "North America"
    elif country_ in Australia_New_Zealand:
        return "Australia and New Zealand"
    elif country_ in Middle_East_Northern_Africa:
        return "Middle East and Northern Africa"
    elif country_ in Latin_America_Caribbean:
        return "Latin America and Caribbean"
    elif country_ in Southeastern_Asia:
        return "Southeastern Asia"
    elif country_ in Central_Eastern_Europe:
        return "Central and Eastern Europe"
    elif country_ in Eastern_Asia:
        return "Eastern Asia"
    elif country_ in Sub_Saharan_Africa:
        return "Sub-Saharan Africa"
    elif country_ in Southern_Asia:
        return "Southern Asia"
    else:
        return "other"

def AddRegion(year):    
    country=year["Country"]
    country=pd.DataFrame(country)
    # country
    # country.info()
    d_country = pd.DataFrame({"Country": list(country["Country"])})
    d_country['Region'] = d_country['Country'].apply(lambda x: FixRegion(x))
    # d_country
    year["Region"]=d_country["Region"]
#     d_2017

AddRegion(d_2017)
AddRegion(d_2018)
AddRegion(d_2019)

display(d_2015.head())
display(d_2016.head())
display(d_2017.head())
display(d_2018.head())
display(d_2019.head())
#we check if there is missing data in our data
# d_2015.isnull().any().any()
# d_2016.isnull().any().any()
# d_2017.isnull().any().any()
# d_2018.isnull().any().any()
# d_2019.isnull().any().any()
# d_2018.dropna() bu 141 adet verimizi 140 a dusereceginden bir tane olan Nan degeri ayni sutunun ortalamasi ile guncellemek gerekir
d_2018["Trust"].fillna((d_2018["Trust"].mean()),inplace=True)
d_2018.info()
def Hap_Score(year):
    region = list(year.Region.unique())
    score = []

    for i in region_list:
        x = year[year.Region == i]
        score.append(x["Score"].mean())


    data = pd.DataFrame({"Region":region,"Score":score})
    data.sort_values("Score",ascending=True,inplace=True)
    return data
  
plt.subplots(1,1)
sns.barplot(x="Score", y="Region", data=Hap_Score(d_2015))
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2015", fontsize=15)


plt.subplots(1,1)
sns.barplot(x="Score", y="Region", data=Hap_Score(d_2016))
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2016", fontsize=15)

plt.subplots(1,1)
sns.barplot(x="Score", y="Region", data=Hap_Score(d_2017))
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2017", fontsize=15)

plt.subplots(1,1)
sns.barplot(x="Score", y="Region", data=Hap_Score(d_2018))
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2018", fontsize=15)

plt.subplots(1,1)
sns.barplot(x="Score", y="Region", data=Hap_Score(d_2019))
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2019", fontsize=15)

plt.show()
d_2015.head()
region = list(d_2015.Region.unique())
score = []
economy = []
family = []
health = []
freedom = []
trust = []

for i in region_list:
    x = d_2015[d_2015.Region == i]
    score.append(x["Score"].mean())
    economy.append(x["GDP"].mean())
    family.append(x["Family"].mean())
    health.append(x["Healthy"].mean())
    freedom.append(x["Freedom"].mean())
    trust.append(x["Trust"].mean())


plt.figure(figsize = (10,5))
sns.barplot(x = economy, y = region, color = "pink", label = "Economy")
sns.barplot(x = family, y = region, color = "red", label = "Family")
sns.barplot(x = health, y = region, color = "blue", label = "Health")
sns.barplot(x = freedom, y = region, color = "orange", label = "Freedom")
sns.barplot(x = trust, y = region, color = "purple", label = "Trust")
plt.legend()
plt.show()
d_2017.head()
region_lists=list(d_2015['Region'].unique())
region_happiness_ratio=[]
region_economy_ratio=[]
for each in region_lists:
    region=d_2015[d_2015['Region']==each]
    region_happiness_rate=sum(region.Score)/len(region)
    region_happiness_ratio.append(region_happiness_rate)
    region_economy_rate=sum(region.GDP)/len(region)
    region_economy_ratio.append(region_economy_rate)
    
data=pd.DataFrame({'region':region_lists,'region_happiness_ratio':region_happiness_ratio,'region_economy_ratio':region_economy_ratio} )
# sorted_data


data['region_happiness_ratio']=data['region_happiness_ratio']/max(data['region_happiness_ratio'])
data['region_economy_ratio']=data['region_economy_ratio']/max(data['region_economy_ratio'])

# data=pd.concat([sorted_data,sorted_data_economy['region_economy_ratio']],axis=1)
# data.sort_values('region_happiness_ratio',inplace=True)

#Visualization
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='region',y='region_happiness_ratio',data=data,color='lime',alpha=0.8)
sns.pointplot(x='region',y='region_economy_ratio',data=data,color='red',alpha=0.8)
plt.text(7.55,0.50,'happiness score ratio',color='red',fontsize = 17,style = 'italic')
plt.text(7.55,0.52,'economy ratio',color='lime',fontsize = 18,style = 'italic')
plt.xticks(rotation=45)
plt.xlabel('Region',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Happiness Score  VS  Economy Rate',fontsize = 20,color='blue')
plt.grid()
plt.show()
f,ax = plt.subplots(figsize=(15, 15))
plt.subplot(5,1,1)
sns.heatmap(d_2015.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f')
plt.title("world happines report-2015 correlation")
plt.xticks(rotation= 30)

plt.subplot(5,1,2)
sns.heatmap(d_2016.corr(), annot=True, linewidths=0.5,linecolor="blue", fmt= '.2f')
plt.title("world happines report-2016 correlation")
plt.xticks(rotation= 30)


plt.subplot(5,1,3)
sns.heatmap(d_2017.corr(), annot=True, linewidths=0.5,linecolor="green", fmt= '.2f')
plt.title("world happines report-2017 correlation")
plt.xticks(rotation= 30)

plt.subplot(5,1,4)
sns.heatmap(d_2017.corr(), annot=True, linewidths=0.5,linecolor="blue", fmt= '.2f')
plt.title("world happines report-2016 correlation")
plt.xticks(rotation= 30)


plt.subplot(5,1,5)
sns.heatmap(d_2018.corr(), annot=True, linewidths=0.5,linecolor="green", fmt= '.2f')
plt.title("world happines report-2017 correlation")
plt.xticks(rotation= 30)

plt.tight_layout()
plt.show()
from plotly.offline import init_notebook_mode,iplot,plot
import plotly.graph_objs as go
import plotly.figure_factory as ff


new_d_2015 = d_2015.loc[:,["GDP","Freedom", "Trust"]]
# new_d_2015
new_d_2015["index"] = np.arange(1,len(new_d_2015)+1)
fig = ff.create_scatterplotmatrix(new_d_2015, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="2015 Trust,Freedom and Economy")
iplot(fig)

new_d_2016 = d_2016.loc[:,["GDP","Freedom", "Trust"]]
new_d_2016["index"] = np.arange(1,len(new_d_2016)+1)
fig = ff.create_scatterplotmatrix(new_d_2016, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="2016 Trust,Freedom and Economy")
iplot(fig)

new_d_2017 = d_2017.loc[:,["GDP","Freedom", "Trust"]]
new_d_2017["index"] = np.arange(1,len(new_d_2017)+1)
fig = ff.create_scatterplotmatrix(new_d_2017, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="2017 Trust,Freedom and Economy")
iplot(fig)

new_d_2018 = d_2018.loc[:,["GDP","Freedom", "Trust"]]
new_d_2018["index"] = np.arange(1,len(new_d_2018)+1)
fig = ff.create_scatterplotmatrix(new_d_2018, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="2018 Trust,Freedom and Economy")
iplot(fig)

new_d_2019 = d_2019.loc[:,["GDP","Freedom", "Trust"]]
new_d_2019["index"] = np.arange(1,len(new_d_2019)+1)
fig = ff.create_scatterplotmatrix(new_d_2019, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="2019 Trust,Freedom and Economy")
iplot(fig)

# x2016 = world_happines_report_2016.Happiness_Score
# x2015 = world_happines_report_2015.Happiness_Score
# x2017 = world_happines_report_2015.Happiness_Score

trace1 = go.Histogram(
    x=d_2015.Score,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=d_2016.Score,
    opacity=0.75,
    name = "2016",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))
trace3 = go.Histogram(
    x=d_2017.Score,
    opacity=0.75,
    name = "2017",
    marker=dict(color='rgba(0, 50, 106, 0.6)'))

trace4 = go.Histogram(
    x=d_2018.Score,
    opacity=0.75,
    name = "2018",
    marker=dict(color='rgba(0, 50, 106, 0.6)'))

trace5 = go.Histogram(
    x=d_2019.Score,
    opacity=0.75,
    name = "2019",
    marker=dict(color='rgba(0, 50, 106, 0.6)'))

data = [trace1, trace2,trace3,trace4,trace5]
layout = go.Layout(barmode='overlay',
                   title='Hapiness Score in 2015,2016,2017,2018 and 2019',
                   xaxis=dict(title='Hapiness Score'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
d_2015.head()
#for 2015
trace1 = go.Scatter3d(
    x=d_2015.Score,
    y=d_2015.GDP,
    z=d_2015.Healthy,
    mode='markers',
    marker=dict(
        color=d_2015.Score,# set color to an array/list of desired values
        colorscale='Portland',             # choose a colorscale
        opacity=0.9,
        size=12,                # set color to an array/list of desired values  
        
    )
)

data = [trace1]
layout = go.Layout(title="2015",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


#for 2016
trace2 = go.Scatter3d(
    x=d_2016.Score,
    y=d_2016.GDP,
    z=d_2016.Healthy,
    mode='markers',
    marker=dict(
        color=d_2016.Score,# set color to an array/list of desired values
        colorscale='Portland',             # choose a colorscale
        opacity=0.9,
        size=12,                # set color to an array/list of desired values  
        
    )
)

data = [trace1]
layout = go.Layout(title="2016",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

#for 2017

trace3 = go.Scatter3d(
    x=d_2017.Score,
    y=d_2017.GDP,
    z=d_2017.Healthy,
    mode='markers',
    marker=dict(
        color=d_2017.Score,# set color to an array/list of desired values
        colorscale='Portland',             # choose a colorscale
        opacity=0.9,
        size=12,                # set color to an array/list of desired values  
        
    )
)

data = [trace1]
layout = go.Layout(title="2017",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

#for 2018

trace4 = go.Scatter3d(
    x=d_2018.Score,
    y=d_2018.GDP,
    z=d_2018.Healthy,
    mode='markers',
    marker=dict(
        color=d_2018.Score,# set color to an array/list of desired values
        colorscale='Portland',             # choose a colorscale
        opacity=0.9,
        size=12,                # set color to an array/list of desired values  
        
    )
)

data = [trace1]
layout = go.Layout(title="2018",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

#for 2019
trace5 = go.Scatter3d(
    x=d_2019.Score,
    y=d_2019.GDP,
    z=d_2019.Healthy,
    mode='markers',
    marker=dict(
        color=d_2019.Score,# set color to an array/list of desired values
        colorscale='Portland',             # choose a colorscale
        opacity=0.9,
        size=12,                # set color to an array/list of desired values  
        
    )
)

data = [trace1]
layout = go.Layout(title="2019",
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
d_2015.head()
import plotly.express as px


fig = px.scatter(d_2015.query("Score>6"), x="GDP", y="Healthy",
         size="Score", color="Region",
                 hover_name="Country", log_x=True, size_max=60, title="Economy, Health and Happiness_Score of Countries which has Happiness Score grather than 5 at 2015 ")
fig.show()

fig = px.scatter(d_2016.query("Score>6"), x="GDP", y="Healthy",
         size="Score", color="Region",
                 hover_name="Country", log_x=True, size_max=60, title="Economy, Health and Happiness_Score of Countries which has Happiness Score grather than 5 at 2016 ")
fig.show()

fig = px.scatter(d_2017.query("Score>6"), x="GDP", y="Healthy",
         size="Score", color="Region",
                 hover_name="Country", log_x=True, size_max=60, title="Economy, Health and Happiness_Score of Countries which has Happiness Score grather than 5 at 2017 ")
fig.show()

fig = px.scatter(d_2018.query("Score>6"), x="GDP", y="Healthy",
         size="Score", color="Region",
                 hover_name="Country", log_x=True, size_max=60, title="Economy, Health and Happiness_Score of Countries which has Happiness Score grather than 5 at 2018 ")
fig.show()

fig = px.scatter(d_2019.query("Score>6"), x="GDP", y="Healthy",
         size="Score", color="Region",
                 hover_name="Country", log_x=True, size_max=60, title="Economy, Health and Happiness_Score of Countries which has Happiness Score grather than 5 at 2019 ")
fig.show()

pie1 = d_2015.iloc[:10,:]['GDP']
labels = d_2015.iloc[:10,:]['Country']
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Economy rates of some Countries",
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Economy rates of first 10 Countries-2015",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Economy Level",
                "x": 0.17,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)


pie1 = d_2016.iloc[:10,:]['GDP']
labels = d_2016.iloc[:10,:]['Country']
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Economy rates of some Countries",
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Economy rates of first 10 Countries-2016",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Economy Level",
                "x": 0.17,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)

pie1 = d_2017.iloc[:10,:]['GDP']
labels = d_2017.iloc[:10,:]['Country']
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Economy rates of some Countries",
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Economy rates of first 10 Countries-2017",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Economy Level",
                "x": 0.17,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)

pie1 = d_2018.iloc[:10,:]['GDP']
labels = d_2018.iloc[:10,:]['Country']
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Economy rates of some Countries",
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Economy rates of first 10 Countries-2018",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Economy Level",
                "x": 0.17,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)

pie1 = d_2019.iloc[:10,:]['GDP']
labels = d_2019.iloc[:10,:]['Country']
# figure
fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Economy rates of some Countries",
      "hoverinfo":"label+percent",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Economy rates of first 10 Countries-2019",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Economy Level",
                "x": 0.17,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  
import warnings            
warnings.filterwarnings("ignore") 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
oku_2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv")
oku_2016=pd.read_csv("/kaggle/input/world-happiness/2016.csv")
oku_2017=pd.read_csv("/kaggle/input/world-happiness/2017.csv")
oku_2018=pd.read_csv("/kaggle/input/world-happiness/2018.csv")
oku_2019=pd.read_csv("/kaggle/input/world-happiness/2019.csv")
display(oku_2015.head(3))
display(oku_2016.head(3))
display(oku_2017.head(3))
display(oku_2018.head(3))
display(oku_2019.head(3))
oku_2015.rename(columns={"Happiness Rank": "Rank","Happiness Score":"Score","Economy (GDP per Capita)":"GDP","Family":"Social support","Health (Life Expectancy)":"Healthy","Trust (Government Corruption)":"Trust"},inplace=True)  
oku_2016.rename(columns={"Happiness Rank": "Rank","Happiness Score":"Score","Economy (GDP per Capita)":"GDP","Family":"Social support","Health (Life Expectancy)":"Healthy","Trust (Government Corruption)":"Trust"},inplace=True)  
oku_2017.rename(columns={"Happiness.Rank": "Rank","Happiness.Score":"Score","Economy..GDP.per.Capita.":"GDP","Family":"Social support","Health..Life.Expectancy.":"Healthy","Trust..Government.Corruption.":"Trust"},inplace=True)  
oku_2018.rename(columns={"Country or region": "Country", "Overall rank": "Rank","Score":"Score","GDP per capita":"GDP","Healthy life expectancy":"Healthy","Freedom to make life choices":"Freedom","Perceptions of corruption":"Trust"},inplace=True) 
oku_2019.rename(columns={"Country or region": "Country", "Overall rank": "Rank","GDP per capita":"GDP","Healthy life expectancy":"Healthy","Freedom to make life choices":"Freedom","Perceptions of corruption":"Trust"},inplace=True)  
display(oku_2015.info())
display(oku_2016.info())
display(oku_2017.info())
display(oku_2018.info())
display(oku_2019.info())
display(oku_2015.head(3))
display(oku_2016.head(3))
display(oku_2017.head(3))
display(oku_2018.head(3))
display(oku_2019.head(3))
def dropit(liste,liste2,liste3,liste4,liste5):
    s=[liste2,liste3,liste4,liste5]
    for i in s:
        for land in liste.index:

            if liste.loc[land].Country not in list(i["Country"]):                  
                    liste.drop(land,inplace=True) 
    
    return liste

oku_2015=dropit(oku_2015,oku_2016,oku_2017,oku_2018,oku_2019)
oku_2015.reset_index(drop=True,inplace=True)

oku_2016=dropit(oku_2016,oku_2015,oku_2017,oku_2018,oku_2019)
oku_2016.reset_index(drop=True,inplace=True)

oku_2017=dropit(oku_2017,oku_2015,oku_2016,oku_2018,oku_2019)
oku_2017.reset_index(drop=True,inplace=True)

oku_2018=dropit(oku_2018,oku_2015,oku_2016,oku_2017,oku_2019)
oku_2018.reset_index(drop=True,inplace=True)

oku_2019=dropit(oku_2019,oku_2016,oku_2017,oku_2018,oku_2015)
oku_2019.reset_index(drop=True,inplace=True)

display(oku_2015.info())
display(oku_2016.info())
display(oku_2017.info())
display(oku_2018.info())
display(oku_2019.info())


def regi_on(stacks):
    stacks.sort_values(by="Country",inplace=True)
    oku_2015.sort_values(by="Country",inplace=True)
    stacks["Region"]=list(oku_2015["Region"])
    stacks=stacks[["Country","Region","Rank","Score","GDP","Social support","Healthy","Freedom","Trust","Generosity"]]
    stacks.sort_values(by="Rank",inplace=True)
    oku_2015.sort_values(by="Rank",inplace=True)
    return stacks
 
oku_2016=regi_on(oku_2016)
oku_2017=regi_on(oku_2017)
oku_2018=regi_on(oku_2018)
oku_2019=regi_on(oku_2019)
oku_2015=regi_on(oku_2015)   

display(oku_2015.head())
display(oku_2016.head())
display(oku_2017.head())
display(oku_2018.head())
display(oku_2019.head())
display(oku_2015.isnull().sum())
display(oku_2016.isnull().sum())
display(oku_2017.isnull().sum())
display(oku_2018.isnull().sum())
display(oku_2019.isnull().sum())
oku_2018.Trust.fillna(oku_2018.Trust.mean(),inplace=True)
display(oku_2018.isnull().sum())




def data_region(stacks):
    score=[i for i in stacks.groupby("Region")["Score"].mean()]
    region=sorted(stacks["Region"].unique())
    dt=pd.DataFrame({"Region":region,"Score":score})
    dt.sort_values(by="Score",inplace=True)
    return dt
plt.subplots(1,1)
sns.barplot(x="Score",y="Region",data=data_region(oku_2015),palette="rocket")
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2015", fontsize=15)


plt.subplots(1,1)
sns.barplot(x="Score",y="Region",data=data_region(oku_2016),palette="rocket")
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2016", fontsize=15)


plt.subplots(1,1)
sns.barplot(x="Score",y="Region",data=data_region(oku_2017),palette="rocket")
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2017", fontsize=15)

plt.subplots(1,1)
sns.barplot(x="Score",y="Region",data=data_region(oku_2018),palette="rocket")
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2018", fontsize=15)

plt.subplots(1,1)
sns.barplot(x="Score",y="Region",data=data_region(oku_2019),palette="rocket")
plt.xlabel("Happiness Score", fontsize=18)
plt.ylabel("Region", fontsize=18)
plt.title("Happiness Score by Region in 2019", fontsize=15)


plt.show()
oku_2015.head(3)
plt.figure(figsize = (10,5))
sns.barplot(x =[i for i in oku_2015.groupby("Region")["GDP"].mean()], 
            y = sorted(oku_2015["Region"].unique()), color = "pink", label = "Economy")
sns.barplot(x = [i for i in oku_2015.groupby("Region")["Social support"].mean()],
            y =sorted(oku_2015["Region"].unique()) , color = "red", label = "Family")
sns.barplot(x = [i for i in oku_2015.groupby("Region")["Healthy"].mean()], 
            y =sorted(oku_2015["Region"].unique()) , color = "blue", label = "Health")
sns.barplot(x = [i for i in oku_2015.groupby("Region")["Freedom"].mean()],
            y = sorted(oku_2015["Region"].unique()), color = "orange", label = "Freedom")
sns.barplot(x = [i for i in oku_2015.groupby("Region")["Trust"].mean()],
            y = sorted(oku_2015["Region"].unique()), color = "purple", label = "Trust")
plt.legend()
plt.show()
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x=sorted(oku_2015["Region"].unique()),y=[i for i in oku_2015.groupby("Region")["Score"].mean()],color='lime',alpha=0.8)
sns.pointplot(x=sorted(oku_2015["Region"].unique()),y=[i*6 for i in oku_2015.groupby("Region")["GDP"].mean()],color='red',alpha=0.8)
sns.pointplot(x=sorted(oku_2015["Region"].unique()),y=[i for i in oku_2015.groupby("Region")["GDP"].mean()],color='black',alpha=0.8)

plt.text(7.55,0.60,'happiness score ratio',color='red',fontsize = 17,style = 'italic')
plt.text(7.55,0.90,'economy ratio',color='lime',fontsize = 18,style = 'italic')
plt.text(7.55,0.30,'real happiness score ratio',color='black',fontsize = 18,style = 'italic')
plt.xticks(rotation=45)
plt.xlabel('Region',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Happiness Score  VS  Economy Rate',fontsize = 20,color='blue')
plt.grid()
plt.show()


f,ax = plt.subplots(figsize=(15, 15))
plt.subplot(5,1,1)
sns.heatmap(oku_2015.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.2f')
plt.title("world happines report-2015 correlation")
plt.xticks(rotation= 30)
print("/n")
plt.subplot(5,1,2)
sns.heatmap(oku_2016.corr(), annot=True, linewidths=0.5,linecolor="blue", fmt= '.2f')
plt.title("world happines report-2016 correlation")
plt.xticks(rotation= 30)
from plotly.offline import init_notebook_mode,iplot,plot
import plotly.graph_objs as go
import plotly.figure_factory as ff


new_d_2015 = oku_2015.loc[:,["GDP","Freedom", "Trust"]]
# new_d_2015
new_d_2015["index"] = np.arange(1,len(new_d_2015)+1)
fig = ff.create_scatterplotmatrix(new_d_2015, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="2015 Trust,Freedom and Economy")
iplot(fig)

new_d_2016 = oku_2016.loc[:,["GDP","Freedom", "Trust"]]
new_d_2016["index"] = np.arange(1,len(new_d_2016)+1)
fig = ff.create_scatterplotmatrix(new_d_2016, diag='box', index='index',colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700,title="2016 Trust,Freedom and Economy")
iplot(fig)
trace1 = go.Histogram(
    x=oku_2015.Score,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=oku_2016.Score,
    opacity=0.75,
    name = "2016",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))
trace3 = go.Histogram(
    x=oku_2017.Score,
    opacity=0.75,
    name = "2017",
    marker=dict(color='rgba(0, 50, 106, 0.6)'))

trace4 = go.Histogram(
    x=oku_2018.Score,
    opacity=0.75,
    name = "2018",
    marker=dict(color='rgba(0, 50, 106, 0.6)'))

trace5 = go.Histogram(
    x=oku_2019.Score,
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
#for 2019
trace1 = go.Scatter3d(
    x=oku_2019.Score,
    y=oku_2019.GDP,
    z=oku_2019.Healthy,
    mode='markers',
    marker=dict(
        color=oku_2019.Score,# set color to an array/list of desired values
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
import plotly.express as px

fig = px.scatter(oku_2019.query("Score>6"), x="GDP", y="Healthy",
         size="Score", color="Region",
                 hover_name="Country", log_x=True, size_max=60, title="Economy, Health and Happiness_Score of Countries which has Happiness Score grather than 5 at 2019 ")
fig.show()
pie1 = oku_2019.iloc[:10,:]['GDP']
labels = oku_2019.iloc[:10,:]['Country']
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

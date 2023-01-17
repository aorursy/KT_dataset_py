import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.tools.plotting import parallel_coordinates
# matplotlib
import matplotlib.pyplot as plt
# seaborn
import seaborn as sns
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
# wordcloud
from wordcloud import WordCloud
# missingno
import missingno as msno
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#loading datas
data_2015 = pd.read_csv("../input/2015.csv")
data_2016 = pd.read_csv("../input/2016.csv")
data_2015.info()
data_2015.head()
data_2016.info()
data_2016.head()
# As we can see, there are no nan values in data_2015
msno.matrix(data_2015)
plt.show()
# There are no nan values in data_2016 too.
msno.matrix(data_2016)
plt.show()
# correlation
f,ax = plt.subplots(figsize=(13,13))
sns.heatmap(data_2015.corr(),annot=True,linewidth=.5,fmt=".1f",ax=ax)
plt.show()
# Regions difference (2015)
# droping columns
rd = data_2015.drop(["Country"],axis=1)
rd = rd.drop(["Happiness Rank"],axis=1)
rd.head()
# visualization
plt.subplots(figsize=(15,15))
parallel_coordinates(rd,"Region",colormap=plt.get_cmap("Set1"))
plt.title("Scores of The Regions")
plt.xlabel("Column")
plt.ylabel("Score")
plt.xticks(rotation=70)
plt.savefig("graph2.png")
plt.show()
# happiness score ratio for region (2015)
region_list15 = list(data_2015.Region.unique())
happiness_score_ratio15 = []

for i in region_list15:
    x = data_2015[data_2015.Region == i]
    happiness_score_ratio15.append(sum(x["Happiness Score"])/len(x))

data = pd.DataFrame({"region_list":region_list15, "happiness_score_ratio":happiness_score_ratio15})
new_index = (data["happiness_score_ratio"].sort_values(ascending=False)).index.values
data15 = data.reindex(new_index)

plt.figure(figsize=(13,13))
sns.barplot(x="region_list", y="happiness_score_ratio", data=data15,
            palette=sns.cubehelix_palette(len(data15.region_list)))
plt.xticks(rotation=90)
plt.xlabel("Region",color="purple")
plt.ylabel("Happiness Score Ratio",color="purple")
plt.title("Happiness Score Ratio for Region",color="purple")
plt.show()
# percentage of Freedom Score (2015)

freedom_score_ratio = []

for i in region_list15:
    a = data_2015[data_2015.Region == i]
    freedom_score_ratio.append(sum(a.Freedom)/len(a))

data2 = pd.DataFrame({"region_list":region_list15, "freedom_score_ratio":freedom_score_ratio})

labels = list(data2.region_list)
ratio = list(data2.freedom_score_ratio)

trace1 = go.Pie(values=ratio,labels=labels,name="Freedom Score",hoverinfo="label+percent+name",hole=.3)
data = [trace1]
layout = dict(title="Percentage of Freedom Score")
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# correlation of economy and happiness (2015)
sns.jointplot(x="Economy (GDP per Capita)", y="Happiness Score", data=data_2015, kind="hex", color="yellow",height=8)
plt.savefig('graph.png')
plt.show()
# correlation of family and freedeom (2015)

sns.kdeplot(data_2015.Family, data_2015.Freedom, shade=True, cut=1,color="red")
plt.title("Correlation of Family and Freedeom",color="red")
plt.xlabel("Family",color="red")
plt.ylabel("Freedeom",color="red")
plt.show()
# scores of health and trust (2015)

data3 = pd.concat([data_2015["Health (Life Expectancy)"],data_2015["Trust (Government Corruption)"]],axis=1)
pal = sns.cubehelix_palette(2,rot=-.4)
sns.violinplot(data=data3, palette=pal, inner="points")
plt.title("Scores of Health and Trust")
plt.show()
# happiness scores by years (2015,2016)

data_2015["Year"] = 2015
data_2016["Year"] = 2016
data4 = pd.concat([data_2015,data_2016],axis=0)
plt.figure(figsize=(13,13))
sns.swarmplot(x="Region", y="Happiness Score", hue="Year", data=data4, size=10)
plt.xticks(rotation=80)
plt.title("Happiness Scores by Years")
plt.xlabel("Region",color="orange")
plt.ylabel("Happiness Score",color="blue")
plt.show()
# number of country in regions that report (2015)

plt.figure(figsize=(12,12))
sns.countplot(data_2015.Region)
plt.xticks(rotation=80)
plt.title("Country Number")
plt.show()
# correlation of happiness and freedom (2015)

data5 = pd.concat([data_2015.Freedom, data_2015["Happiness Score"]],axis=1)
sns.pairplot(data5)
plt.show()
# region's economy scores  by years (2015,2016)

plt.figure(figsize=(15,13))
sns.boxplot(x="Region", y="Economy (GDP per Capita)", hue="Year", data=data4)
plt.xticks(rotation=70)
plt.title("Region's Economy Scores by Years")
plt.show()
# correlation of generosity and freedom by region (2015)

f,ax1 = plt.subplots(figsize=(15,10))
sns.pointplot(x="Region", y="Generosity", data=data_2015, color="blue",alpha=0.8)
sns.pointplot(x="Region", y="Freedom", data=data_2015, color="yellow",alpha=0.5)
plt.text(8,0.6,"Generosity",color="blue",fontsize=18)
plt.text(8,0.55,"Freedom",color="yellow",fontsize=18)
plt.xticks(rotation=80)
plt.ylabel("Score")
plt.grid()
plt.show()
# happiness rank and happiness score for countries by years (2015,2016)

trace1 = go.Scatter(x=data_2015["Happiness Rank"],
                    y=data_2015["Happiness Score"],
                    mode="lines",
                    name="2015",
                    marker=dict(color="rgba(0,255,0,0.8)"),
                    text=data_2015.Country)

trace2 = go.Scatter(x=data_2016["Happiness Rank"],
                    y=data_2016["Happiness Score"],
                    mode="lines+markers",
                    name="2016",
                    marker=dict(color="rgba(150,0,70,0.8)"),
                    text=data_2016.Country)

trace_data = [trace1,trace2]
layout = dict(title="Happiness Rank and Happiness Score for Countries by Years",
              xaxis=dict(title="Happiness Rank",ticklen=.5,zeroline=False),
              yaxis=dict(title="Happiness Score",ticklen=.5,zeroline=False))
fig = dict(data=trace_data, layout=layout)
iplot(fig)
# country economy scores by happiness rank (2015,2016)

trace1 = go.Scatter(x=data_2015["Happiness Rank"],
                    y=data_2015["Economy (GDP per Capita)"],
                    mode="markers",
                    name="2015",
                    marker=dict(color="rgba(255, 128, 255, 0.8)"),
                    text=data_2015.Country)
trace2 = go.Scatter(x=data_2016["Happiness Rank"],
                    y=data_2016["Economy (GDP per Capita)"],
                    mode="markers",
                    name="2016",
                    marker=dict(color="rgba(0, 255, 200, 0.8)"),
                    text=data_2016.Country)
trace_data = [trace1,trace2]
layout = dict(title="Country Economy Scores by Happiness Rank",
              xaxis=dict(title="Happiness Rank",ticklen=.5,zeroline=False),
              yaxis=dict(title="Economy (GDP per Capita)",ticklen=.5,zeroline=False))
fig = dict(data=trace_data, layout=layout)
iplot(fig)
# health and trust scores by top 3 country of the most happy (2015)

top3 = data_2015.iloc[:3,:]
trace1 = go.Bar(x=top3.Country,
                y=top3["Health (Life Expectancy)"],
                name="Health (Life Expectancy)",
                text=top3.Region)

trace2 = go.Bar(x=top3.Country,
                y=top3["Trust (Government Corruption)"],
                name="Trust (Government Corruption)",
                text=top3.Region)
data = [trace1,trace2]
layout = dict(title="Health and Trust Scores",barmode="relative",xaxis=dict(title="Top Happy 3 Country"))
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# wordcloud 2016
plt.subplots(figsize=(10,10))
wc = WordCloud(background_color="black",width=550,height=450).generate(" ".join(data_2016.Country))
plt.imshow(wc)
plt.axis("off")
plt.show()
# 3d scatter plot (2016)
# x= happines rank, y=happiness score, z=freedom

trace1 = go.Scatter3d(x=data_2016["Happiness Rank"],
                      y=data_2016["Happiness Score"],
                      z=data_2016["Freedom"],
                      mode="markers",
                      marker=dict(color=data_2016["Happiness Rank"],size=12,colorscale="Greens"),
                      text=data_2016.Country)
data = [trace1]
layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
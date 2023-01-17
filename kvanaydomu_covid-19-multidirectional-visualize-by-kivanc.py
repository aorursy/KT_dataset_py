countryWise["Deaths / 100 Recovered"]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.graph_objs as go
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
countryWise = pd.read_csv("/kaggle/input/corona-virus-report/country_wise_latest.csv")
countryWise.head()
countryWise.columns
countryWise["WHO Region"]
countryWise.describe()
countryWise.info()
countryWise["Deaths / 100 Cases"]=countryWise["Deaths / 100 Cases"].astype("int")
countryWise["Recovered / 100 Cases"]=countryWise["Recovered / 100 Cases"].astype("int")
def barPlotVirus(variable):
    """
    input : variable : "Active"
    output : bar plot & value count
    """
    var = countryWise[variable]
    varValue = var.value_counts()
    plt.figure(figsize=(9,5))
    plt.bar(varValue.index,varValue.values)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{} : \n {}".format(variable,varValue))
    
    
category=["WHO Region"]
for c in category:
    barPlotVirus(c)
def plotHistVirus(variable):
    plt.figure(figsize=(9,3))
    plt.hist(countryWise[variable],bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} variable showing with hist".format(variable))
    plt.show()
    
numericVar=["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths", "New recovered", "Deaths / 100 Cases", "Recovered / 100 Cases", "Confirmed last week","1 week change", "1 week % increase"]
for n in numericVar:
    plotHistVirus(n)
# WHO Region-Confirmed
countryWise[["WHO Region","Confirmed"]].groupby(["WHO Region"],as_index=False).mean().sort_values(by="Confirmed",ascending=False)
# WHO Region-Deaths
countryWise[["WHO Region","Deaths"]].groupby(["WHO Region"],as_index=False).mean().sort_values(by="Deaths",ascending=False)
# WHO Region-Recovered
countryWise[["WHO Region","Recovered"]].groupby(["WHO Region"],as_index=False).mean().sort_values(by="Recovered",ascending=False)
# WHO Region-Active
countryWise[["WHO Region","Active"]].groupby(["WHO Region"],as_index=False).mean().sort_values(by="Active",ascending=False)
# WHO Region-1 week change
countryWise[["WHO Region","1 week change"]].groupby(["WHO Region"],as_index=False).mean().sort_values(by="1 week change",ascending=False)
# WHO Region-1 week % increase
countryWise[["WHO Region","1 week % increase"]].groupby(["WHO Region"],as_index=False).mean().sort_values(by="1 week % increase",ascending=False)
# WHO Region-Deaths / 100 Cases
countryWise[["WHO Region","1 week change"]].groupby(["WHO Region"],as_index=False).mean().sort_values(by="1 week change",ascending=False)
def detectOutliers(df,features):
    outlier_indices=[]
    for c in features:
        Q1=np.percentile(df[c],25)
        Q2=np.percentile(df[c],75)
        IQR = Q2-Q1
        outlierStep = IQR*1.5
        outlierListCol = df[(df[c] < Q1-outlierStep) | (df[c]>Q2+outlierStep)].index
        outlier_indices.extend(outlierListCol)
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)
    return multiple_outliers
countryWise.loc[detectOutliers(countryWise,["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths", "New recovered", "Deaths / 100 Cases", "Recovered / 100 Cases", "Confirmed last week","1 week change", "1 week % increase"])]
# countryWise=countryWise.drop(detectOutliers(countryWise,["Confirmed", "Deaths", "Recovered", "Active", "New cases", "New deaths", "New recovered", "Deaths / 100 Cases", "Recovered / 100 Cases", "Confirmed last week","1 week change", "1 week % increase"]),axis=0).reset_index(drop=True)
countryWise.columns[countryWise.isnull().any()]
list1=["Confirmed","Deaths","Recovered","Active","1 week change","1 week % increase"]
plt.figure(figsize=(10,5))
sns.heatmap(countryWise[list1].corr(),annot=True,fmt=".2f")
plt.show()
g=sns.factorplot(x="WHO Region",y="Confirmed",data=countryWise,kind="bar",size=6)
g.set_ylabels("Confirmed rates")
g.add_legend()
plt.xticks(rotation=45)
plt.title("Compare Regions According to Confirmed Rates")
plt.show()
sortingConfirmedIndex=countryWise[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
SortConfirmed=countryWise.reindex(sortingConfirmedIndex)
HastMostConfirmed=SortConfirmed.head(10)
#HastMostConfirmed
sns.factorplot(x="Country/Region",y="Confirmed",data=HastMostConfirmed,kind="bar",size=6)
g.set_ylabels("Confirmed rates")
g.add_legend()
plt.xticks(rotation=45)
plt.title("Compare Countries That Has Most Confirmed People All Over The World")
plt.show()
europeConfirmed=countryWise[countryWise["WHO Region"]=="Europe"]
europeConfIndex=europeConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
europeConfirmed=europeConfirmed.reindex(europeConfIndex)
topTenEurope=europeConfirmed.head(10)
values=topTenEurope["Confirmed"]
labels=topTenEurope["Country/Region"]

data = {
    "values":values,
    "labels":labels,
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Europe That Has Most Confirmed on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
europeConfirmed=countryWise[countryWise["WHO Region"]=="Europe"]
europeConfIndex=europeConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
europeConfirmed=europeConfirmed.reindex(europeConfIndex)
topTenEurope=europeConfirmed.head(10)

data = {
    "values":topTenEurope["Confirmed"],
    "labels":topTenEurope["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Europe That Has Most Confirmed on Virus"
}
fig=dict(data=data,layout=layout)
iplot(fig)
americasConfirmed=countryWise[countryWise["WHO Region"]=="Americas"]
americasConfIndex=americasConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
americasConfirmed=americasConfirmed.reindex(americasConfIndex)
topTenAmerica=americasConfirmed.head(10)
data = {
    "values":topTenAmerica["Confirmed"],
    "labels":topTenAmerica["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Americas That Has Most Confirmed on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
africaConfirmed=countryWise[countryWise["WHO Region"]=="Africa"]
africaConfIndex=africaConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
africaConfirmed=africaConfirmed.reindex(africaConfIndex)
topTenAfrica=africaConfirmed.head(10)
data = {
    "values":topTenAfrica["Confirmed"],
    "labels":topTenAfrica["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Africa That Has Most Confirmed on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
easternConfirmed=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
easternConfIndex=easternConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
easternConfirmed=easternConfirmed.reindex(easternConfIndex)
topTenEastern=easternConfirmed.head(10)
data = {
    "values":topTenEastern["Confirmed"],
    "labels":topTenEastern["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Eastern Mediterranean That Has Most Confirmed on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
westernConfirmed=countryWise[countryWise["WHO Region"]=="Western Pacific"]
westernConfIndex=westernConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
westernConfirmed=westernConfirmed.reindex(westernConfIndex)
topTenWestern=westernConfirmed.head(10)
data = {
    "values":topTenWestern["Confirmed"],
    "labels":topTenWestern["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Western Pacific That Has Most Confirmed on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
SouthAsiaConfirmed=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthAsiaConfIndex=SouthAsiaConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
SouthAsiaConfirmed=SouthAsiaConfirmed.reindex(SouthAsiaConfIndex)
topTenSouthAsia=SouthAsiaConfirmed.head(10)
data = {
    "values":topTenSouthAsia["Confirmed"],
    "labels":topTenSouthAsia["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In South-East Asia That Has Most Confirmed on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
g=sns.factorplot(x="WHO Region",y="Deaths",data=countryWise,kind="bar",size=6)
g.set_ylabels("Deaths rates")
plt.xticks(rotation=45)
plt.title("Compare Regions According to Death Rates")
g.add_legend()
plt.show()
sortingDeathsIndex=countryWise[["Deaths"]].sort_values(by="Deaths",ascending=False).index
sortDeaths = countryWise.reindex(sortingDeathsIndex)
HasMostDeaths = sortDeaths.head(10)
sns.factorplot(x="Country/Region",y="Deaths",data=HasMostDeaths,kind="bar",size=6)
g.set_ylabels("Deaths rates")
g.add_legend()
plt.xticks(rotation=45)
plt.title("Comparing Top 10 Countries That Has Most Deaths from Epidemic")
plt.show()
europeDeaths=countryWise[countryWise["WHO Region"]=="Europe"]
europeDeadIndex=europeDeaths[["Deaths"]].sort_values(by="Deaths",ascending=False).index
europeDeaths=countryWise.reindex(europeDeadIndex)
topTenEurope=europeDeaths.head(10)
data = {
    "values":topTenEurope["Deaths"],
    "labels":topTenEurope["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Europe That Has Most Deaths on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
europeDeaths
AfricaDeaths=countryWise[countryWise["WHO Region"]=="Africa"]
AfricaDeadIndex=AfricaDeaths[["Deaths"]].sort_values(by="Deaths",ascending=False).index
AfricaConfirmed=AfricaDeaths.reindex(AfricaDeadIndex)
topTenAfrica=AfricaDeaths.head(10)
data = {
    "values":topTenAfrica["Deaths"],
    "labels":topTenAfrica["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Africa That Has Most Deaths on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
AmericasDeaths=countryWise[countryWise["WHO Region"]=="Americas"]
AmericasDeadIndex=AmericasDeaths[["Deaths"]].sort_values(by="Deaths",ascending=False).index
AmericasDeaths=AmericasDeaths.reindex(AmericasDeadIndex)
topTenAmericas=AmericasDeaths.head(10)
data = {
    "values":topTenAmericas["Deaths"],
    "labels":topTenAmericas["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Americas That Has Most Deaths on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
EasternDeaths=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
EasternDeadIndex=EasternDeaths[["Deaths"]].sort_values(by="Deaths",ascending=False).index
EasternDeaths=EasternDeaths.reindex(EasternDeadIndex)
topTenEastern=EasternDeaths.head(10)
data = {
    "values":topTenEastern["Deaths"],
    "labels":topTenEastern["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Eastern Mediterranean That Has Most Deaths on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
WesternDeaths=countryWise[countryWise["WHO Region"]=="Western Pacific"]
WesternDeadIndex=WesternDeaths[["Deaths"]].sort_values(by="Deaths",ascending=False).index
WesternDeaths=WesternDeaths.reindex(WesternDeadIndex)
topTenWestern=WesternDeaths.head(10)
data = {
    "values":topTenWestern["Deaths"],
    "labels":topTenWestern["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Western Pacific That Has Most Deaths on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
SouthAsiaDeaths=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthAsiaDeadIndex=SouthAsiaDeaths[["Deaths"]].sort_values(by="Deaths",ascending=False).index
SouthAsiaDeaths=SouthAsiaDeaths.reindex(SouthAsiaDeadIndex)
topTenSouthAsia=SouthAsiaDeaths.head(10)
data = {
    "values":topTenSouthAsia["Deaths"],
    "labels":topTenSouthAsia["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In South-East Asia That Has Most Deaths on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
g=sns.factorplot(x="WHO Region",y="Recovered",data=countryWise,kind="bar",size=6)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Compare Regions According to Recovered Rates")
plt.show()
sortingRecoveredIndex=countryWise[["Recovered"]].sort_values(by="Recovered",ascending=False).index
sortingRecoveredRates=countryWise.reindex(sortingRecoveredIndex)
recoveredTopTen=sortingRecoveredRates.head(10)
g=sns.factorplot(x="Country/Region",y="Recovered",data=recoveredTopTen,kind="bar",size=6)
plt.xticks(rotation=45)
g.add_legend()
plt.title("Top 10 Countries That Has Most Recovered Peoples Counts")
plt.show()
europeRecovered=countryWise[countryWise["WHO Region"]=="Europe"]
europeRecoveredIndex=europeRecovered[["Recovered"]].sort_values(by="Recovered",ascending=False).index
europeRecovered=europeRecovered.reindex(europeRecoveredIndex)
topTenEurope=europeRecovered.head(10)
data = {
    "values":topTenEurope["Recovered"],
    "labels":topTenEurope["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Europe That Has Most Recovered on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
AmericasRecovered=countryWise[countryWise["WHO Region"]=="Americas"]
AmericasRecoveredIndex=AmericasRecovered[["Recovered"]].sort_values(by="Recovered",ascending=False).index
AmericasRecovered=AmericasRecovered.reindex(AmericasRecoveredIndex)
topAmericas=AmericasRecovered.head(10)
data = {
    "values":topAmericas["Recovered"],
    "labels":topAmericas["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Americas That Has Most Recovered on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
AfricaRecovered=countryWise[countryWise["WHO Region"]=="Africa"]
AfricaRecoveredIndex=AfricaRecovered[["Recovered"]].sort_values(by="Recovered",ascending=False).index
AfricaRecovered=AfricaRecovered.reindex(AfricaRecoveredIndex)
topAfrica=AfricaRecovered.head(10)
data = {
    "values":topAfrica["Recovered"],
    "labels":topAfrica["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Africa That Has Most Recovered on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
EasternRecovered=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
EasternRecoveredIndex=EasternRecovered[["Recovered"]].sort_values(by="Recovered",ascending=False).index
AfricaRecovered=EasternRecovered.reindex(EasternRecoveredIndex)
topEastern=EasternRecovered.head(10)
data = {
    "values":topEastern["Recovered"],
    "labels":topEastern["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Eastern Mediterranean That Has Most Recovered on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
WesternRecovered=countryWise[countryWise["WHO Region"]=="Western Pacific"]
WesternRecoveredIndex=WesternRecovered[["Recovered"]].sort_values(by="Recovered",ascending=False).index
AfricaRecovered=WesternRecovered.reindex(WesternRecoveredIndex)
topWestern=WesternRecovered.head(10)
data = {
    "values":topWestern["Recovered"],
    "labels":topWestern["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Western Pacific That Has Most Recovered on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
SouthRecovered=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthRecoveredIndex=SouthRecovered[["Recovered"]].sort_values(by="Recovered",ascending=False).index
AfricaRecovered=SouthRecovered.reindex(SouthRecoveredIndex)
topSouth=SouthRecovered.head(10)
data = {
    "values":topSouth["Recovered"],
    "labels":topSouth["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In South-East Asia That Has Most Recovered on Virus"
}
fig=go.Figure(data=data,layout=layout)
iplot(fig)
g=sns.factorplot(x="WHO Region",y="Active",data=countryWise,kind="bar",size=6)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Compare Regions According to Active Rates")
plt.show()
activeSortIndex=countryWise[["Active"]].sort_values(by="Active",ascending=False).index
active=countryWise.reindex(activeSortIndex).head(10)
g=sns.factorplot(x="Country/Region",y="Active",data=active,kind="bar",size=6)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Top 10 Countries That Has Most Active Patients")
plt.show()
europeActive = countryWise[countryWise["WHO Region"]=="Europe"]
europeActiveIndex = europeActive[["Active"]].sort_values(by="Active",ascending=False).index
europeActive=countryWise.reindex(europeActiveIndex)
topActive=europeActive.head(10)


data = {
    "values":topActive["Recovered"],
    "labels":topActive["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Europe That Has Most Active on Virus"
}

figure=go.Figure(data=data,layout=layout)
iplot(figure)
AfricaActive = countryWise[countryWise["WHO Region"]=="Africa"]
AfricaActiveIndex = AfricaActive[["Active"]].sort_values(by="Active",ascending=False).index
AfricaActive=countryWise.reindex(AfricaActiveIndex)
topActiveAfrica=AfricaActive.head(10)


data = {
    "values":topActiveAfrica["Recovered"],
    "labels":topActiveAfrica["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Africa That Has Most Active on Virus"
}

figure=go.Figure(data=data,layout=layout)
iplot(figure)
AmericasActive = countryWise[countryWise["WHO Region"]=="Americas"]
AmericasActiveIndex = AmericasActive[["Active"]].sort_values(by="Active",ascending=False).index
AmericasActive=countryWise.reindex(AmericasActiveIndex)
topActiveAmericas=AmericasActive.head(10)


data = {
    "values":topActiveAmericas["Recovered"],
    "labels":topActiveAmericas["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Americas That Has Most Active on Virus"
}

figure=go.Figure(data=data,layout=layout)
iplot(figure)
EasternActive = countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
EasternActiveIndex = EasternActive[["Active"]].sort_values(by="Active",ascending=False).index
EasternActive=countryWise.reindex(EasternActiveIndex)
topActiveEastern=EasternActive.head(10)


data = {
    "values":topActiveEastern["Recovered"],
    "labels":topActiveEastern["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Eastern Mediterranean That Has Most Active on Virus"
}

figure=go.Figure(data=data,layout=layout)
iplot(figure)
WesternActive = countryWise[countryWise["WHO Region"]=="Western Pacific"]
WesternActiveIndex = WesternActive[["Active"]].sort_values(by="Active",ascending=False).index
WesternActive=countryWise.reindex(WesternActiveIndex)
topActiveWestern=WesternActive.head(10)


data = {
    "values":topActiveWestern["Recovered"],
    "labels":topActiveWestern["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In Western Pacific That Has Most Active on Virus"
}

figure=go.Figure(data=data,layout=layout)
iplot(figure)
SouthActive = countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthActiveIndex = SouthActive[["Active"]].sort_values(by="Active",ascending=False).index
SouthnActive=countryWise.reindex(SouthActiveIndex)
topActiveSouth=SouthnActive.head(10)


data = {
    "values":topActiveSouth["Recovered"],
    "labels":topActiveSouth["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In South-East Asia That Has Most Active on Virus"
}

figure=go.Figure(data=data,layout=layout)
iplot(figure)
SouthActive = countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthActiveIndex = SouthActive[["Active"]].sort_values(by="Active",ascending=False).index
SouthnActive=countryWise.reindex(SouthActiveIndex)
topActiveSouth=SouthnActive.head(10)


data = {
    "values":topActiveSouth["Recovered"],
    "labels":topActiveSouth["Country/Region"],
    "type":"pie",
    "hoverinfo":"label+percent",
    "hole":.3
}
layout={
    "title":"Top 10 Countries In South-East Asia That Has Most Active on Virus"
}

figure=go.Figure(data=data,layout=layout)
iplot(figure)
mostConfirmedIndex=countryWise[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
mostConfirmed=countryWise.reindex(mostConfirmedIndex).head(10)
labels=mostConfirmed["Country/Region"]
values=mostConfirmed["1 week % increase"]
confirmedRates=mostConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    name="1 week % increase",
    marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmedRates
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week % increase rates of Countries That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
europeConfirmed=countryWise[countryWise["WHO Region"]=="Europe"]
europeConfIndex=europeConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
europeConfirmed=europeConfirmed.reindex(europeConfIndex).head(10)


labels=europeConfirmed["Country/Region"]
values=europeConfirmed["1 week % increase"]
confirmed=europeConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week % increase rates of Countries in Europe That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
americasConfirmed=countryWise[countryWise["WHO Region"]=="Americas"]
americasConfIndex=americasConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
americasConfirmed=americasConfirmed.reindex(americasConfIndex).head(10)

labels=americasConfirmed["Country/Region"]
values=americasConfirmed["1 week % increase"]
confirmed=americasConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week % increase rates of Countries in America That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
africaConfirmed=countryWise[countryWise["WHO Region"]=="Africa"]
africaConfIndex=africaConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
africaConfirmed=africaConfirmed.reindex(africaConfIndex).head(10)

labels=africaConfirmed["Country/Region"]
values=africaConfirmed["1 week % increase"]
confirmed=africaConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week % increase rates of Countries in Africa That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
easternConfirmed=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
easternConfIndex=easternConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
easternConfirmed=easternConfirmed.reindex(easternConfIndex).head(10)

labels=easternConfirmed["Country/Region"]
values=easternConfirmed["1 week % increase"]
confirmed=easternConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week % increase rates of Countries in Eastern Mediterranean That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
westernConfirmed=countryWise[countryWise["WHO Region"]=="Western Pacific"]
westernConfIndex=westernConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
westernConfirmed=westernConfirmed.reindex(westernConfIndex).head(10)

labels=westernConfirmed["Country/Region"]
values=westernConfirmed["1 week % increase"]
confirmed=westernConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week % increase of Countries in Western Pacific That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
SouthAsiaConfirmed=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthAsiaConfIndex=SouthAsiaConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
SouthAsiaConfirmed=SouthAsiaConfirmed.reindex(SouthAsiaConfIndex).head(10)

labels=SouthAsiaConfirmed["Country/Region"]
values=SouthAsiaConfirmed["1 week % increase"]
confirmed=SouthAsiaConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week % increase of Countries in South-East Asia That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
mostConfirmedIndex=countryWise[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
mostConfirmed=countryWise.reindex(mostConfirmedIndex).head(10)
labels=mostConfirmed["Country/Region"]
values=mostConfirmed["1 week change"]
confirmed=mostConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week change rates of Countries That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
europeConfirmed=countryWise[countryWise["WHO Region"]=="Europe"]
europeConfIndex=europeConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
europeConfirmed=europeConfirmed.reindex(europeConfIndex).head(10)


labels=europeConfirmed["Country/Region"]
values=europeConfirmed["1 week change"]
confirmed=europeConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week change rates of Countries in Europe That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
americasConfirmed=countryWise[countryWise["WHO Region"]=="Americas"]
americasConfIndex=americasConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
americasConfirmed=americasConfirmed.reindex(americasConfIndex).head(10)

labels=americasConfirmed["Country/Region"]
values=americasConfirmed["1 week change"]
confirmed=americasConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week change rates of Countries in America That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
africaConfirmed=countryWise[countryWise["WHO Region"]=="Africa"]
africaConfIndex=africaConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
africaConfirmed=africaConfirmed.reindex(africaConfIndex).head(10)

labels=africaConfirmed["Country/Region"]
values=africaConfirmed["1 week change"]
confirmed=africaConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week change rates of Countries in Africa That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
easternConfirmed=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
easternConfIndex=easternConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
easternConfirmed=easternConfirmed.reindex(easternConfIndex).head(10)

labels=easternConfirmed["Country/Region"]
values=easternConfirmed["1 week change"]
confirmed=easternConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week change rates of Countries in Eastern Mediterranean That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
westernConfirmed=countryWise[countryWise["WHO Region"]=="Western Pacific"]
westernConfIndex=westernConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
westernConfirmed=westernConfirmed.reindex(westernConfIndex).head(10)

labels=westernConfirmed["Country/Region"]
values=westernConfirmed["1 week change"]
confirmed=westernConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week change of Countries in Western Pacific That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
SouthAsiaConfirmed=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthAsiaConfIndex=SouthAsiaConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
SouthAsiaConfirmed=SouthAsiaConfirmed.reindex(SouthAsiaConfIndex).head(10)

labels=SouthAsiaConfirmed["Country/Region"]
values=SouthAsiaConfirmed["1 week change"]
confirmed=SouthAsiaConfirmed["Confirmed"]
trace1=go.Bar(
    x=labels,
    y=values,
    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)
data=[trace1]
layout = go.Layout(barmode="group",title="1 week change of Countries in South-East Asia That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
correlation=countryWise[["Confirmed","Deaths","Recovered"]]
sns.heatmap(correlation.corr(),fmt=".2f",annot=True)
plt.title("Correlation Between Recovered,Deaths and Confirmed")
plt.show()
mostConfirmedIndex=countryWise[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
mostConfirmed=countryWise.reindex(mostConfirmedIndex).head(10)
labels=mostConfirmed["Country/Region"]
deaths = mostConfirmed["Deaths"]
recovered=mostConfirmed["Recovered"]
confirmed=mostConfirmed["Confirmed"]

trace1=go.Bar(
    x=labels,
    y=deaths,
    name="Deaths",
    marker = dict(color = 'rgba(255, 174, 255, 0.5)',                          
    line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)


trace2=go.Bar(
    x=labels,
    y=recovered,
    name="Recovered",
    marker=dict(color='rgba(0, 204, 204, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)

data=[trace1,trace2];
layout=dict(barmode="group",title="Recovered and Deaths Rates Belong To Countries That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
europeConfirmed=countryWise[countryWise["WHO Region"]=="Europe"]
europeConfIndex=europeConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
europeConfirmed=europeConfirmed.reindex(europeConfIndex).head(10)
labels=europeConfirmed["Country/Region"]
deaths=europeConfirmed["Deaths"]
recovered=europeConfirmed["Recovered"]
confirmed=europeConfirmed["Confirmed"]

trace1=go.Bar(
    x=labels,
    y=deaths,
    name="Deaths",
    marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    text=deaths
)


trace2=go.Bar(
    x=labels,
    y=recovered,
    name="Recovered",
    marker=dict(color='rgba(0, 204, 204, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=recovered
)

data=[trace1,trace2];
layout=dict(barmode="group",title="Recovered and Deaths Rates Belong To Countries in Europe That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
americasConfirmed=countryWise[countryWise["WHO Region"]=="Americas"]
americasConfIndex=americasConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
americasConfirmed=americasConfirmed.reindex(americasConfIndex).head(10)

labels=americasConfirmed["Country/Region"]
deaths=americasConfirmed["Deaths"]
recovered=americasConfirmed["Recovered"]
confirmed=americasConfirmed["Confirmed"]

trace1=go.Bar(
    x=labels,
    y=deaths,
    name="Deaths",
    marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)


trace2=go.Bar(
    x=labels,
    y=recovered,
    name="Recovered",
    marker=dict(color='rgba(0, 204, 204, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)

data=[trace1,trace2];
layout=dict(barmode="group",title="Recovered and Deaths Rates Belong To Countries in America That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
africaConfirmed=countryWise[countryWise["WHO Region"]=="Africa"]
africaConfIndex=africaConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
africaConfirmed=africaConfirmed.reindex(africaConfIndex).head(10)

labels=africaConfirmed["Country/Region"]
deaths=africaConfirmed["Deaths"]
recovered=africaConfirmed["Recovered"]
confirmed=africaConfirmed["Confirmed"]

trace1=go.Bar(
    x=labels,
    y=deaths,
    name="Deaths",
    marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)


trace2=go.Bar(
    x=labels,
    y=recovered,
    name="Recovered",
    marker=dict(color='rgba(0, 204, 204, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)

data=[trace1,trace2];
layout=dict(barmode="group",title="Recovered and Deaths Rates Belong To Countries in Africa That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
easternConfirmed=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
easternConfIndex=easternConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
easternConfirmed=easternConfirmed.reindex(easternConfIndex).head(10)

labels=easternConfirmed["Country/Region"]
deaths=easternConfirmed["Deaths"]
recovered=easternConfirmed["Recovered"]
confirmed=easternConfirmed["Confirmed"]

trace1=go.Bar(
    x=labels,
    y=deaths,
    name="Deaths",
    marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)


trace2=go.Bar(
    x=labels,
    y=recovered,
    name="Recovered",
    marker=dict(color='rgba(0, 204, 204, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)

data=[trace1,trace2];
layout=dict(barmode="group",title="Recovered and Deaths Rates Belong To Countries in Eastern Mediterranean That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
westernConfirmed=countryWise[countryWise["WHO Region"]=="Western Pacific"]
westernConfIndex=westernConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
westernConfirmed=westernConfirmed.reindex(westernConfIndex).head(10)

labels=westernConfirmed["Country/Region"]
deaths=westernConfirmed["Deaths"]
recovered=westernConfirmed["Recovered"]
confirmed=westernConfirmed["Confirmed"]

trace1=go.Bar(
    x=labels,
    y=deaths,
    name="Deaths",
    marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)


trace2=go.Bar(
    x=labels,
    y=recovered,
    name="Recovered",
    marker=dict(color='rgba(0, 204, 204, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)

data=[trace1,trace2];
layout=dict(barmode="group",title="Recovered and Deaths Rates Belong To Countries in Western Pacific That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)

SouthAsiaConfirmed=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthAsiaConfIndex=SouthAsiaConfirmed[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
SouthAsiaConfirmed=SouthAsiaConfirmed.reindex(SouthAsiaConfIndex).head(10)


labels=SouthAsiaConfirmed["Country/Region"]
deaths=SouthAsiaConfirmed["Deaths"]
recovered=SouthAsiaConfirmed["Recovered"]
confirmed=SouthAsiaConfirmed["Confirmed"]

trace1=go.Bar(
    x=labels,
    y=deaths,
    name="Deaths",
    marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)


trace2=go.Bar(
    x=labels,
    y=recovered,
    name="Recovered",
    marker=dict(color='rgba(0, 204, 204, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),
    text=confirmed
)

data=[trace1,trace2];
layout=dict(barmode="group",title="Recovered and Deaths Rates Belong To Countries in South-East Asia That Has Most Confirmed Patients")
fig=go.Figure(data=data,layout=layout)
iplot(fig)
HasMostDeathsIndex=countryWise[["Deaths"]].sort_values(by="Deaths",ascending=False).index
HasMostDeaths=countryWise.reindex(HasMostDeathsIndex).head(10)
Deaths=HasMostDeaths["Deaths"]
Recovered=HasMostDeaths["Recovered"]
OneWeekIncrease=HasMostDeaths["1 week % increase"]
OneWeekChange=HasMostDeaths["1 week change"]
labels=HasMostDeaths["Country/Region"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]
layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries That Recovered and One week % increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]
layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries That Recovered and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
europe=countryWise[countryWise["WHO Region"]=="Europe"]
europeDeathsIndex=europe[["Deaths"]].sort_values(by="Deaths",ascending=False).index
europeMostDeaths=europe.reindex(europeDeathsIndex).head(10)
labels=europeMostDeaths["Country/Region"]
Recovered=europeMostDeaths["Recovered"]
OneWeekIncrease=europeMostDeaths["1 week % increase"]
OneWeekChange=europeMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Europe That Recovered and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Europe That Recovered and One week % Increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
america=countryWise[countryWise["WHO Region"]=="Americas"]
americaDeathsIndex=america[["Deaths"]].sort_values(by="Deaths",ascending=False).index
americaMostDeaths=america.reindex(americaDeathsIndex).head(10)
labels=americaMostDeaths["Country/Region"]
Recovered=americaMostDeaths["Recovered"]
OneWeekIncreaseamericaMostDeaths=americaMostDeaths["1 week % increase"]
OneWeekChange=americaMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in America That Recovered and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
america = countryWise[countryWise["WHO Region"]=="Americas"]
americasIndex = america[["Deaths"]].sort_values(by="Deaths",ascending=False).index
americaTop=america.reindex(americasIndex).head(10)
labels = americaTop["Country/Region"]
recovered = americaTop["Recovered"]
OneWeekIncrease=americaTop["1 week % increase"]

trace1 = go.Scatter(
    x=labels,
    y=recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)

data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
         anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in America That Recovered and One week increase"
)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
Africa=countryWise[countryWise["WHO Region"]=="Africa"]
AfricaaDeathsIndex=Africa[["Deaths"]].sort_values(by="Deaths",ascending=False).index
AfricaMostDeaths=Africa.reindex(AfricaaDeathsIndex).head(10)
labels=AfricaMostDeaths["Country/Region"]
Recovered=AfricaMostDeaths["Recovered"]
OneWeekIncrease=AfricaMostDeaths["1 week % increase"]
OneWeekChange=AfricaMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Africa That Recovered and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Africa=countryWise[countryWise["WHO Region"]=="Africa"]
AfricaaDeathsIndex=Africa[["Deaths"]].sort_values(by="Deaths",ascending=False).index
AfricaMostDeaths=Africa.reindex(AfricaaDeathsIndex).head(10)
labels=AfricaMostDeaths["Country/Region"]
Recovered=AfricaMostDeaths["Recovered"]
OneWeekIncrease=AfricaMostDeaths["1 week % increase"]
OneWeekChange=AfricaMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Africa That Recovered and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Eastern=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
EasternDeathsIndex=Eastern[["Deaths"]].sort_values(by="Deaths",ascending=False).index
EasternMostDeaths=Eastern.reindex(EasternDeathsIndex).head(10)
labels=EasternMostDeaths["Country/Region"]
Recovered=EasternMostDeaths["Recovered"]
OneWeekIncrease=EasternMostDeaths["1 week % increase"]
OneWeekChange=EasternMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Eastern Mediterranean That Recovered and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Western=countryWise[countryWise["WHO Region"]=="Western Pacific"]
WesternDeathsIndex=Western[["Deaths"]].sort_values(by="Deaths",ascending=False).index
WesternMostDeaths=Western.reindex(WesternDeathsIndex).head(10)
labels=WesternMostDeaths["Country/Region"]
Recovered=WesternMostDeaths["Recovered"]
OneWeekIncrease=WesternMostDeaths["1 week % increase"]
OneWeekChange=WesternMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Eastern Mediterranean That Recovered and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Western=countryWise[countryWise["WHO Region"]=="Western Pacific"]
WesternDeathsIndex=Western[["Deaths"]].sort_values(by="Deaths",ascending=False).index
WesternMostDeaths=Western.reindex(WesternDeathsIndex).head(10)
labels=WesternMostDeaths["Country/Region"]
Recovered=WesternMostDeaths["Recovered"]
OneWeekIncrease=WesternMostDeaths["1 week % increase"]
OneWeekChange=WesternMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Eastern Mediterranean That Recovered and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
South=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthDeathsIndex=South[["Deaths"]].sort_values(by="Deaths",ascending=False).index
SouthMostDeaths=South.reindex(SouthDeathsIndex).head(10)
labels=SouthMostDeaths["Country/Region"]
Recovered=SouthMostDeaths["Recovered"]
OneWeekIncrease=SouthMostDeaths["1 week % increase"]
OneWeekChange=SouthMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Eastern Mediterranean That Recovered and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
South=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthDeathsIndex=South[["Deaths"]].sort_values(by="Deaths",ascending=False).index
SouthMostDeaths=South.reindex(SouthDeathsIndex).head(10)
labels=SouthMostDeaths["Country/Region"]
Recovered=SouthMostDeaths["Recovered"]
OneWeekIncrease=SouthMostDeaths["1 week % increase"]
OneWeekChange=SouthMostDeaths["1 week change"]

trace1=go.Scatter(
    x=labels,
    y=Recovered,
    name="Recovered",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=OneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="OneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.6, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Deaths Countries in Eastern Mediterranean That Recovered and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
recoveredDivCasesIndex=countryWise[["Recovered / 100 Cases"]].sort_values(by="Recovered / 100 Cases",ascending=False).index
recoveredDivCasesTop = countryWise.reindex(recoveredDivCasesIndex).head(20)
recoveredDivCasesBad = countryWise.reindex(recoveredDivCasesIndex).tail(20)
recoveredDivCasesBad

g=sns.factorplot(x="Country/Region",y="Recovered / 100 Cases",data=recoveredDivCasesTop,kind="bar",size=8)
g.add_legend()
plt.xticks(rotation=45)
plt.title("RECOVERED PEOPLE RATES /100 CASES ACCORDING TO TOP 20 SUCCESS COUNTRIES")
plt.show()
g=sns.factorplot(x="Country/Region",y="Recovered / 100 Cases",data=recoveredDivCasesBad,kind="bar",size=9)
g.add_legend()
plt.xticks(rotation=45)
plt.title("RECOVERED PEOPLE RATES /100 CASES ACCORDING TO TOP 20 INEFFECTIVE COUNTRIES")
plt.show()
countryWise.columns
europe=countryWise[countryWise["WHO Region"]=="Europe"]
europeConfirmedIndex=europe[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
europeMostConfirmed=europe.reindex(europeConfirmedIndex).head(15)
labels=europeMostConfirmed["Country/Region"]
oneWeekChange=europeMostConfirmed["1 week change"]
oneWeekIncrease=europeMostConfirmed["1 week % increase"]
recoveredDivCases=europeMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.7, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.7, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Europe That Recovered / 100 Cases and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.7, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.7, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Europe That Recovered / 100 Cases and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Americas=countryWise[countryWise["WHO Region"]=="Americas"]
AmericasConfirmedIndex=Americas[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
AmericasMostConfirmed=Americas.reindex(AmericasConfirmedIndex).head(15)
labels=AmericasMostConfirmed["Country/Region"]
oneWeekChange=AmericasMostConfirmed["1 week change"]
oneWeekIncrease=AmericasMostConfirmed["1 week % increase"]
recoveredDivCases=AmericasMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.7, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.7, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Americas That Recovered / 100 Cases and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.7, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.7, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in America That Recovered / 100 Cases and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Africa=countryWise[countryWise["WHO Region"]=="Africa"]
AfricaConfirmedIndex=Africa[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
AfricaMostConfirmed=Africa.reindex(AfricaConfirmedIndex).head(15)
labels=AfricaMostConfirmed["Country/Region"]
oneWeekChange=AfricaMostConfirmed["1 week change"]
oneWeekIncrease=AfricaMostConfirmed["1 week % increase"]
recoveredDivCases=AfricaMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.7, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.9, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Africa That Recovered / 100 Cases and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Africa=countryWise[countryWise["WHO Region"]=="Africa"]
AfricaConfirmedIndex=Africa[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
AfricaMostConfirmed=Africa.reindex(AfricaConfirmedIndex).head(15)
labels=AfricaMostConfirmed["Country/Region"]
oneWeekChange=AfricaMostConfirmed["1 week change"]
oneWeekIncrease=AfricaMostConfirmed["1 week % increase"]
recoveredDivCases=AfricaMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.7, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.9, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Africa That Recovered / 100 Cases and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Eastern=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
EasternConfirmedIndex=Eastern[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
EasternMostConfirmed=Eastern.reindex(EasternConfirmedIndex).head(15)
labels=EasternMostConfirmed["Country/Region"]
oneWeekChange=EasternMostConfirmed["1 week change"]
oneWeekIncrease=EasternMostConfirmed["1 week % increase"]
recoveredDivCases=EasternMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.5, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.9, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Eastern Mediterranean That Recovered / 100 Cases and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Eastern=countryWise[countryWise["WHO Region"]=="Eastern Mediterranean"]
EasternConfirmedIndex=Eastern[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
EasternMostConfirmed=Eastern.reindex(EasternConfirmedIndex).head(15)
labels=EasternMostConfirmed["Country/Region"]
oneWeekChange=EasternMostConfirmed["1 week change"]
oneWeekIncrease=EasternMostConfirmed["1 week % increase"]
recoveredDivCases=EasternMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
   xaxis2=dict(
        domain=[0.5, 0.95],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.9, 0.95],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Eastern Mediterranean That Recovered / 100 Cases and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Western=countryWise[countryWise["WHO Region"]=="Western Pacific"]
WesternConfirmedIndex=Western[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
WesternMostConfirmed=Western.reindex(WesternConfirmedIndex).head(15)
labels=WesternMostConfirmed["Country/Region"]
oneWeekChange=WesternMostConfirmed["1 week change"]
oneWeekIncrease=WesternMostConfirmed["1 week % increase"]
recoveredDivCases=WesternMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.80],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.5, 0.70],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Western Pacific That Recovered / 100 Cases and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
Western=countryWise[countryWise["WHO Region"]=="Western Pacific"]
WesternConfirmedIndex=Western[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
WesternMostConfirmed=Western.reindex(WesternConfirmedIndex).head(15)
labels=WesternMostConfirmed["Country/Region"]
oneWeekChange=WesternMostConfirmed["1 week change"]
oneWeekIncrease=WesternMostConfirmed["1 week % increase"]
recoveredDivCases=WesternMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
   xaxis2=dict(
        domain=[0.6, 0.80],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.5, 0.70],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in Western Pacific That Recovered / 100 Cases and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
South=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthConfirmedIndex=South[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
SouthMostConfirmed=South.reindex(SouthConfirmedIndex).head(15)
labels=SouthMostConfirmed["Country/Region"]
oneWeekChange=SouthMostConfirmed["1 week change"]
oneWeekIncrease=SouthMostConfirmed["1 week % increase"]
recoveredDivCases=SouthMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekChange,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekChange",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
 xaxis2=dict(
        domain=[0.6, 0.80],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.5, 0.70],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in South-East Asia That Recovered / 100 Cases and One week change"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
South=countryWise[countryWise["WHO Region"]=="South-East Asia"]
SouthConfirmedIndex=South[["Confirmed"]].sort_values(by="Confirmed",ascending=False).index
SouthMostConfirmed=South.reindex(SouthConfirmedIndex).head(15)
labels=SouthMostConfirmed["Country/Region"]
oneWeekChange=SouthMostConfirmed["1 week change"]
oneWeekIncrease=SouthMostConfirmed["1 week % increase"]
recoveredDivCases=SouthMostConfirmed["Recovered / 100 Cases"]

trace1=go.Scatter(
    x=labels,
    y=recoveredDivCases,
    name="recoveredDivCases",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)

trace2=go.Scatter(
    x=labels,
    y=oneWeekIncrease,
    xaxis='x2',
    yaxis='y2',
    name="oneWeekIncrease",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
)


data = [trace1, trace2]

layout=go.Layout(
  xaxis2=dict(
        domain=[0.6, 0.80],
        anchor='y2'
    ),
    yaxis2=dict(
         domain=[0.5, 0.70],
        anchor='x2'
    ),
    title="Show Both Rates of Has Most Confirmed Countries in South-East Asia That Recovered / 100 Cases and One week increase"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
countryWise.columns
deathsDivCasesIndex = countryWise[["Deaths / 100 Cases"]].sort_values(by="Deaths / 100 Cases",ascending=False).index
deathsDivCases=countryWise.reindex(deathsDivCasesIndex).head(20)
g=sns.factorplot(x="Country/Region",y="Deaths / 100 Cases",data=deathsDivCases,kind="bar",size=10)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Top 20 Countries Rates That Has Deaths in 100 Cases")
plt.show()
deathsDivCasesEur=countryWise[countryWise["WHO Region"] == "Europe"]
deathsDivCasesEurIndex=deathsDivCasesEur[["Deaths / 100 Cases"]].sort_values(by="Deaths / 100 Cases",ascending=False).index
deathsDivCasesEur=countryWise.reindex(deathsDivCasesEurIndex).head(20)

g=sns.factorplot(x="Country/Region",y="Deaths / 100 Cases",data=deathsDivCasesEur,kind="bar",size=10)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Top 20 Countries in Europe Rates That Has Deaths in 100 Cases")
plt.show()
deathsDivCasesAfrica=countryWise[countryWise["WHO Region"] == "Africa"]
deathsDivCasesAfricaIndex=deathsDivCasesAfrica[["Deaths / 100 Cases"]].sort_values(by="Deaths / 100 Cases",ascending=False).index
deathsDivCasesAfrica=countryWise.reindex(deathsDivCasesAfricaIndex).head(20)

g=sns.factorplot(x="Country/Region",y="Deaths / 100 Cases",data=deathsDivCasesAfrica,kind="bar",size=10)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Top 20 Countries in Africa Rates That Has Deaths in 100 Cases")
plt.show()
deathsDivCasesAmericas=countryWise[countryWise["WHO Region"] == "Americas"]
deathsDivCasesAmericasIndex=deathsDivCasesAmericas[["Deaths / 100 Cases"]].sort_values(by="Deaths / 100 Cases",ascending=False).index
deathsDivCasesAmericas=countryWise.reindex(deathsDivCasesAmericasIndex).head(20)

g=sns.factorplot(x="Country/Region",y="Deaths / 100 Cases",data=deathsDivCasesAmericas,kind="bar",size=10)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Top 20 Countries in Africa Rates That Has Deaths in 100 Cases")
plt.show()
deathsDivCasesEastern=countryWise[countryWise["WHO Region"] == "Eastern Mediterranean"]
deathsDivCasesEasternIndex=deathsDivCasesEastern[["Deaths / 100 Cases"]].sort_values(by="Deaths / 100 Cases",ascending=False).index
deathsDivCasesEastern=countryWise.reindex(deathsDivCasesEasternIndex).head(20)

g=sns.factorplot(x="Country/Region",y="Deaths / 100 Cases",data=deathsDivCasesEastern,kind="bar",size=10)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Top 20 Countries in Eastern Mediterranean Rates That Has Deaths in 100 Cases")
plt.show()
deathsDivCasesWestern=countryWise[countryWise["WHO Region"] == "Western Pacific"]
deathsDivCasesWesternIndex=deathsDivCasesWestern[["Deaths / 100 Cases"]].sort_values(by="Deaths / 100 Cases",ascending=False).index
deathsDivCasesWestern=countryWise.reindex(deathsDivCasesWesternIndex).head(20)

g=sns.factorplot(x="Country/Region",y="Deaths / 100 Cases",data=deathsDivCasesWestern,kind="bar",size=10)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Top 20 Countries in Western Pacific Rates That Has Deaths in 100 Cases")
plt.show()
deathsDivCasesSouth=countryWise[countryWise["WHO Region"] == "South-East Asia"]
deathsDivCasesSouthIndex=deathsDivCasesSouth[["Deaths / 100 Cases"]].sort_values(by="Deaths / 100 Cases",ascending=False).index
deathsDivCasesSouth=countryWise.reindex(deathsDivCasesSouthIndex).head(20)

g=sns.factorplot(x="Country/Region",y="Deaths / 100 Cases",data=deathsDivCasesSouth,kind="bar",size=10)
g.add_legend()
plt.xticks(rotation=45)
plt.title("Top 20 Countries in South-East Asia Rates That Has Deaths in 100 Cases")
plt.show()
countryWise["WHO Region"].unique()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

import plotly.graph_objs as go

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
googlePlay = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

googlePlay.head()
googlePlay.columns
googlePlay.describe() # one columns that has numbers data
googlePlay.Price.value_counts()
googlePlay.Type.value_counts()
googlePlay["Content Rating"]
googlePlay.Genres
googlePlay["Current Ver"]
googlePlay["Android Ver"]
googlePlay.drop_duplicates(subset="App",inplace=True)

googlePlay=googlePlay[googlePlay["Android Ver"] != np.nan]

googlePlay=googlePlay[googlePlay["Android Ver"] != "NaN"]

googlePlay=googlePlay[googlePlay["Installs"] != "Free"]

googlePlay=googlePlay[googlePlay["Installs"] != "Paid"]
googlePlay["Installs"] = googlePlay["Installs"].apply(lambda x : x.replace("+","") if "+" in str(x) else x)

googlePlay["Installs"] = googlePlay["Installs"].apply(lambda x : x.replace(",","") if "," in str(x) else x)

googlePlay["Installs"] = googlePlay["Installs"].apply(lambda x : int(x))
googlePlay["Size"]=googlePlay["Size"].apply(lambda x : str(x).replace("Varies with device","NaN") if "Varies with device" in str(x) else x)

googlePlay["Size"] = googlePlay["Size"].apply(lambda x : str(x).replace("M","") if "M" in str(x) else x)

googlePlay["Size"]=googlePlay["Size"].apply(lambda x : str(x).replace(",","") if "," in str(x) else x)

googlePlay["Size"] = googlePlay["Size"].apply(lambda x : float(str(x).replace("k",""))/1000 if "k" in str(x) else x)
googlePlay["Size"]=googlePlay["Size"].apply(lambda x : float(x))

googlePlay["Installs"] = googlePlay["Installs"].apply(lambda x : float(x))
googlePlay["Price"] = googlePlay["Price"].apply(lambda x : str(x).replace("$","") if "$" in str(x) else str(x))

googlePlay["Price"]=googlePlay["Price"].apply(lambda x : float(x))
googlePlay["Reviews"] = googlePlay["Reviews"].apply(lambda x : int(x))
googlePlay.columns[googlePlay.isnull().any()]
googlePlay["Rating"].dropna(inplace=True)

googlePlay["Size"].dropna(inplace=True)

googlePlay["Type"].dropna(inplace=True)

googlePlay["Current Ver"].dropna(inplace=True)

googlePlay["Android Ver"].dropna(inplace=True)
Counter(googlePlay["Rating"].isnull())

Counter(googlePlay["Size"].isnull())

Counter(googlePlay["Type"].isnull())
googlePlay.info()
def barPlotApps(variable):

    var = googlePlay[variable]

    varValue = var.value_counts()

    plt.figure(figsize=(14,5))

    plt.bar(varValue.index,varValue.values)

    plt.xticks(varValue.index,rotation=90)

    plt.ylabel("Frequency")

    plt.title("Frequency of according to categories")

    plt.show()

    print("{} : \n {}".format(variable,varValue))

    
category = "Category"



googlePlay[category].value_counts().index
category = ["Category"]



for c in category:

    barPlotApps(c)
def plotHistApp(variable):

    plt.figure(figsize=(14,5))

    plt.hist(googlePlay[variable],bins=50)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} variable showing with hist".format(variable))

    plt.show()
numericVar = ['Rating', 'Reviews','Installs','Size','Price']



for n in numericVar:

    plotHistApp(n)
googlePlay.columns
# Category - Rating

googlePlay[["Category","Rating"]].groupby(["Category"],as_index=False).mean().sort_values(by="Rating",ascending=False)
# Category - Reviews

googlePlay.Reviews=googlePlay.Reviews.replace(np.nan,0)

googlePlay[["Category","Reviews"]].groupby(["Category"],as_index=False).mean().sort_values(by="Reviews",ascending=False)
#Category - Installs

googlePlay.Installs=googlePlay.Installs.replace(np.nan,0)

googlePlay[["Category","Installs"]].groupby(["Category"],as_index=False).mean().sort_values(by="Installs",ascending=False)
# Category - Price

googlePlay[["Category","Price"]].groupby(["Category"],as_index=False).mean().sort_values(by="Price",ascending=False)
# Content Rating - Reviews



googlePlay[["Content Rating","Reviews"]].groupby(["Content Rating"],as_index=False).mean().sort_values(by="Reviews",ascending=False)
# Content Rating - Installs



googlePlay[["Content Rating","Installs"]].groupby(["Content Rating"],as_index=False).mean().sort_values(by="Installs",ascending=False)
# Content Rating - Price

googlePlay[["Content Rating","Price"]].groupby(["Content Rating"],as_index=False).mean().sort_values(by="Price",ascending=False)
# App - Rating

googlePlay.Rating = googlePlay.Rating.replace(np.nan,0)

googlePlay[["App","Rating"]].groupby(["App"],as_index=False).mean().sort_values(by="Rating",ascending=False)
# App - Reviews



googlePlay[["App","Reviews"]].groupby(["App"],as_index=False).mean().sort_values(by="Reviews",ascending=False)
# App - Installs  # million

googlePlay[["App","Installs"]].groupby(["App"],as_index=False).mean().sort_values(by="Installs",ascending=False)
# App - Price

googlePlay.Price=googlePlay.Price.replace(np.nan,0)

googlePlay[["App","Price"]].groupby(["App"],as_index=False).mean().sort_values(by="Price",ascending=False)
def findOutliers(data,columns):

    outlierIndices = []

    for c in columns:

        Q1 = np.percentile(data[c],25)

        Q2 = np.percentile(data[c],75)

        IQR = Q2-Q1

        outlierStep = IQR*1.5

        outlierListIndexes = data[(data[c] < Q1-outlierStep) | (data[c] > Q2+outlierStep)].index

        outlierIndices.extend(outlierListIndexes)

    outlierIndices=Counter(outlierIndices)

    multiOutliers = list(i for i,v in outlierIndices.items() if v>2)

    return multiOutliers
googlePlay.loc[findOutliers(googlePlay,["Rating","Reviews","Size","Installs","Price"])]
numeriColumns = ["Category", "Rating" ,"Reviews","Size","Installs"]

#googlePlay.loc[:,numeriColumns]

plt.figure(figsize=(10,7))

sns.heatmap(googlePlay[numeriColumns].corr(),annot=True,fmt="0.5f")

plt.title("Correlation Between Columns")

plt.show()
categoryIndex=googlePlay["Category"].value_counts().index

categoryValues=googlePlay["Category"].value_counts().values



data = {

    "values" : categoryValues,

    "labels" : categoryIndex,

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}



layout = {

    "title" : "Frequency of categories inside of data"

}



figure=go.Figure(data=data,layout=layout)

iplot(figure)
topInstallIndex = googlePlay[["Installs"]].sort_values(by="Installs",ascending=False).index

topInstall=googlePlay.reindex(topInstallIndex).head(100)



g=sns.factorplot(x="Category",y="Installs",data=topInstall,kind="bar",size=10)

g.add_legend()

plt.xticks(rotation=90)

g.set_ylabels("Category Frequency according to Install")

plt.show()
topRatingIndex = googlePlay[["Rating"]].sort_values(by="Rating",ascending=False).index

topRating=googlePlay.reindex(topRatingIndex).head(100)



g=sns.factorplot(x="Category",y="Rating",data=topRating,kind="bar",size=10)

g.add_legend()

plt.xticks(rotation=90)

g.set_ylabels("Category Frequency according to Rating")

plt.show()
topReviewsIndex = googlePlay[["Reviews"]].sort_values(by="Reviews",ascending=False).index

topReviews=googlePlay.reindex(topReviewsIndex).head(30)



labels = topReviews["Category"]

installs = topReviews["Installs"]

reviews = topReviews["Reviews"]



trace1=go.Bar(

    x = labels,

    y=reviews,

    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),

    text = installs

)



data=trace1



layout = go.Layout(barmode="group",title="Relationship Category,Installs,Reviews according to Categories which has more reviews",xaxis={"title":"Categories"},yaxis={"title":"Reviews"})

fig = go.Figure(data=data,layout=layout)

iplot(fig)
typeIndex=googlePlay.Type.value_counts().index

typeValues = googlePlay.Type.value_counts().values
plt.subplots(figsize=(10,10))

sns.barplot(x=typeIndex,y=typeValues,color="blue")

plt.xlabel("Type")

plt.ylabel("Counts")

plt.show()
googlePlay.head()
topInstallsIndex = googlePlay[["Installs"]].sort_values(by="Installs",ascending=False).index

topInstalls=googlePlay.reindex(topInstallsIndex)
GroupByType=topInstalls.groupby(["Type"],as_index=False).mean()

GroupByType


#%%

data = {

    "values" : GroupByType["Installs"],

    "labels" : GroupByType["Type"],

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}

layout = {

    "title" : "According to Type , Rates of Installs"

}



fig=go.Figure(data=data,layout=layout)

iplot(fig)
data = {

    "values" : GroupByType["Reviews"],

    "labels" : GroupByType["Type"],

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}

layout = {

    "title" : "According to Type , Rates of Reviews"

}



fig=go.Figure(data=data,layout=layout)

iplot(fig)
data = {

    "values" : GroupByType["Rating"],

    "labels" : GroupByType["Type"],

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}

layout = {

    "title" : "According to Type , Rates of Rating"

}



fig=go.Figure(data=data,layout=layout)

iplot(fig)
categyAndType=googlePlay[["Category","Type"]].groupby(["Type"],as_index=False).count()

categyAndType
data = {

    "values" : categyAndType["Category"],

    "labels" : categyAndType["Type"],

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}

layout = {

    "title" : "According to Categories Rates Type"

}



fig=go.Figure(data=data,layout=layout)

iplot(fig)
TopTenRateIndex = googlePlay[["Rating"]].sort_values(by="Rating",ascending=False).index

TopTenRate=googlePlay.reindex(TopTenRateIndex).head(10)

TopTenRate

g=sns.factorplot(x="App",y="Rating",data=TopTenRate,kind="bar",size=8,)

g.set_ylabels("Rating")

g.add_legend()

plt.xticks(rotation=90)

plt.show()
TopTenReviewsIndex = googlePlay[["Reviews"]].sort_values(by="Reviews",ascending=False).index

TopTenReviews=googlePlay.reindex(TopTenReviewsIndex).head(30)

g=sns.factorplot(x="App",y="Reviews",data=TopTenReviews,kind="bar",size=10)

g.add_legend()

plt.xticks(rotation=90)

g.set_ylabels("Reviews counts(b)")

plt.show()
googlePlay.Installs
topInstallIndex = googlePlay[["Installs"]].sort_values(by="Installs",ascending=False).index

topInstall = googlePlay.reindex(topInstallIndex).head(30)

g=sns.factorplot(x="App",y="Installs",data=topInstall,kind="bar",size=10)

g.add_legend()

g.set_ylabels("Installs count(billion)")

plt.xticks(rotation=90)

plt.show()
pieInstall = topInstall.head(10)

 

data = {

    "values" : pieInstall["Installs"],

    "labels" : pieInstall["Category"],

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}



layout = {

    "title" : "Show Relationship Categories and installs"

}



fig=go.Figure(data=data,layout=layout)

iplot(fig)
googlePlay.head()
topInstallsIndex = googlePlay[["Installs"]].sort_values(by="Installs",ascending=False).index

topInstalls=googlePlay.reindex(topInstallsIndex)


data = {

    "values" : topInstalls["Installs"],

    "labels" : topInstalls["Content Rating"],

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}



layout = {

    "title" : "Show Relationship Content Rating and installs"

}



fig=go.Figure(data=data,layout=layout)

iplot(fig)
TopTenRateIndex = googlePlay[["Rating"]].sort_values(by="Rating",ascending=False).index

TopTenRate=googlePlay.reindex(TopTenRateIndex)



data = {

    "values" : TopTenRate["Rating"],

    "labels" : TopTenRate["Content Rating"],

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}



layout = {

    "title" : "Show Relationship Content Rating and installs"

}



fig=go.Figure(data=data,layout=layout)

iplot(fig)
TopTenReviewseIndex = googlePlay[["Reviews"]].sort_values(by="Reviews",ascending=False).index

TopTenReviews=googlePlay.reindex(TopTenReviewseIndex)



data = {

    "values" : TopTenReviews["Reviews"],

    "labels" : TopTenReviews["Content Rating"],

    "type" : "pie",

    "hoverinfo" : "label+percent",

    "hole" : .3

}



layout = {

    "title" : "Show Relationship Content Reviews and installs"

}



fig=go.Figure(data=data,layout=layout)

iplot(fig)
labels = topInstall["App"]

installs = topInstall["Installs"]

rating = topInstall["Rating"]



trace1=go.Bar(

    x = labels,

    y=installs,

    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),

    text = rating

)



data=trace1



layout = go.Layout(barmode="group",title="Relationship App,Installs,Rating according to Top 30 Apps which more download")

fig = go.Figure(data=data,layout=layout)

iplot(fig)
labels = topInstall["App"]

installs = topInstall["Installs"]

reviews = topInstall["Reviews"]



trace1=go.Bar(

    x = labels,

    y=installs,

    marker=dict(color='rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),

    text = reviews

)



data=trace1



layout = go.Layout(barmode="group",title="Relationship App,Installs,Reviews according to Top 30 Apps which more download")

fig = go.Figure(data=data,layout=layout)

iplot(fig)
topInstall.head()
print("Averoage of price app on google play = {}".format(googlePlay["Price"].mean()))
plt.subplots(figsize=(15,10))

sns.lineplot(x="Price",y="Installs",data=googlePlay,color="red")

plt.show()
plt.subplots(figsize=(15,10))

sns.lineplot(x="Price",y="Rating",data=googlePlay,color="green")

plt.title("Relationship Price & Rating")

plt.show()
plt.subplots(figsize=(15,10))

sns.lineplot(x="Price",y="Reviews",data=googlePlay,color="green")

plt.title("Relationship Price & Reviews")

plt.show()
categoryPrices=googlePlay.groupby(["Category"],as_index=False).mean()

g=sns.factorplot(x="Category",y="Price",kind="bar",data=categoryPrices,size=10)

g.add_legend()

g.set_ylabels("Prices Rates")

plt.xticks(rotation=90)

plt.show()
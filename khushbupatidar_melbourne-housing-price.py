# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the rquired library



import pandas as pd

import numpy as np

import datetime

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



import cufflinks as cf

cf.go_offline()

# load the required data



home_data=pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")

home_data.head() # observations
# shape(Number of rows and columns) of the data



home_data.shape
home_data.info()
print("Categorical variable : ", home_data.select_dtypes(include=["O"]).columns.to_list())
print("Numerical variable : ", home_data.select_dtypes(exclude=["O"]).columns.to_list())
# Lets describe the datasets

home_data.describe().T
# Convert the "Postcode" into categorical variable



home_data["Postcode"]=home_data["Postcode"].astype("category")



home_data.info()


# check for duplicate columns.Datasets have "Rooms" and "Bedroom2" features which could be duplicate.



home_data["room"]=home_data["Rooms"]-home_data["Bedroom2"]



home_data
# drop the selected columns



home_data.drop(["Bedroom2","room"],axis=1,inplace=True)

home_data.columns
# Check for missing value(counts)



home_data.isnull().sum().sort_values(ascending=False)
# percentage of missing value



home_data.isnull().sum()/len(home_data)*100
# Handle the missing value



# fill the missing value in "Car" variable



home_data.fillna(value={"Car": 0},inplace=True)



# Drop the missing value in other column



home_data.dropna(inplace=True)

home_data.info()

home_data.describe().T
#  house with BuildingArea==0 



home_data[home_data["BuildingArea"] ==0]

# Drop the rows where BuildingArea is 0 which is not possible for any house to be 0 size. It acts as an outlier. 



home_data=home_data[home_data["BuildingArea"] !=0]
# House with Landsize = 0

home_data[home_data["Landsize"]==0]
# Analyse the price variable



plt.style.use("ggplot")

plt.figure(figsize=(10,6))



sns.distplot(home_data["Price"],kde=False,hist_kws=dict(edgecolor="k"))

plt.title("House Price Distribution In Melbourne",size=16);
## Log transformation of price variable 

plt.style.use("classic")

plt.figure(figsize=(10,6))

sns.distplot(np.log(home_data["Price"]),kde=False)



plt.title("Distribution of Log Tranformed Price ", size=16);
home_data.columns
# Price analysis based on suburb 

# Prepare the data



suburb_Price=home_data.groupby("Suburb",as_index=False)["Price"].mean().sort_values(by="Price",ascending=False).reset_index(drop=True)

suburb_Price.rename(columns={"Price":"AveragePrice"},inplace=True)

suburb_Price.head(20)
# Top 20 costliest Suburb





fig=px.bar(suburb_Price.head(20),x="AveragePrice",y="Suburb",color="Suburb",title="Top 20 Costliest Suburb",text="AveragePrice",orientation="h",height=800,width=900)

fig.update_traces(textposition="inside")

fig.update_layout(plot_bgcolor='rgb(193,255,193)')

fig.show()
# Top 20 least costlier Suburb



fig=px.bar(suburb_Price.tail(20),x="AveragePrice",y="Suburb",color="Suburb",title="Top 20 Least Costliest Suburb",text="AveragePrice",orientation="h",height=800,width=900)

fig.update_traces(textposition="inside")

fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')

fig.show()
# Rooms vs Suburb

# Prepare the data



rooms_suburb=home_data.groupby("Suburb")[["Rooms","Price"]].mean().sort_values(by="Price",ascending=False).reset_index()

rooms_suburb.head()
# Average number of rooms available per house in Costliest Suburb



rooms_suburb[["Suburb","Rooms"]].head(20).iplot(kind="bar",x="Suburb",title="Average Number of Rooms ")
# Property count Vs Price



fig=px.scatter(home_data,x="Propertycount",y="Price",color="Type",title="Price Distribution Vs PropertyCount")

fig.show()
# Analyse the price as per Availability of the number of rooms

# Prepare the data



room_data=home_data.groupby("Rooms")["Price"].mean().sort_values(ascending=False).reset_index()

room_data



plt.figure(figsize=(14,6))

sns.boxplot(x="Rooms",y="Price",data=home_data)

plt.title('Price analysis Vs Rooms',size=16);
# Rooms Vs Price along with regression 



sns.lmplot(x="Rooms",y="Price",hue="Type",data=home_data)

plt.title("Price Vs Rooms",size=15);
# Convert the Date variable into datetime object



home_data["Date"]=pd.to_datetime(home_data["Date"])



# add Year column to home_data

home_data["Year"]=home_data.Date.dt.year

# Price Distibution Over last 10 year



year_data=home_data[home_data["YearBuilt"]>2008]



fig=px.box(year_data,x="YearBuilt",y="Price",title="Price Distribution Over Last 10 Years")

fig.show()
# Different Regionname and which region are preferred more 



home_data.Regionname.value_counts()
# Price of the house based on the different region



home_data.groupby("Regionname")["Price"].mean().sort_values(ascending=False)
# 

plt.figure(figsize=(14,7))



sns.boxplot(x="Regionname",y="Price",data=home_data)

plt.xticks(rotation=45)

plt.title("Price Distribution Over Different Region",size=15);

# Analyse Price against Type of house

print("Type of the houses in melbourne : ",home_data.Type.value_counts().count())
# 

print("Count of h type house  : ",home_data.Type.value_counts()[0])

print("Count of u type house  : ",home_data.Type.value_counts()[1])

print("Count of t type house  : ",home_data.Type.value_counts()[2])
plt.figure(figsize=(8,6))

sns.set(style="darkgrid")

sns.countplot(x="Type",data=home_data)

plt.title("Type of House in Melbourne",size=15);
# Lets explore how price varyies with different type of house

print("Average Price for the h - house,cottage,villa, semi,terrace : $%.f "  % home_data.groupby("Type").Price.mean()[0])

print("Average Price for the u - unit, duplex  : $%.f " % home_data.groupby("Type").Price.mean()[2])

print("Average Price for the t - townhouse : $%.f " % home_data.groupby("Type").Price.mean()[1])
# price Distribution across Type of house



fig=px.box(home_data,y="Type",x="Price",title="Price Distribution Vs Type of House",orientation="h")

fig.show()
# Price Distribution Over last 10 Year Based On Type of House



fig=px.box(year_data,y="Price",x="YearBuilt",color="Type",title="Price Distribution Over last 10 Year Based On Type of House")

fig.show()
print("Number of property Sold: ",home_data.Method.value_counts()[0])

print("Number of property Sold prior: ",home_data.Method.value_counts()[1])

print("Number of property Passed in: ",home_data.Method.value_counts()[2])

print("Number of Vendor Bid: ",home_data.Method.value_counts()[3])

print("Number of property Sold after auction: ",home_data.Method.value_counts()[4])

# Analysis based on the property sold

 # Prepare the data

df=home_data[(home_data["Method"]!="PI")]

df1=df[(df["Method"]!="VB")]
# Number of property sold 

plt.figure(figsize=(8,6))

sns.set(style="darkgrid")

sns.countplot(x="Method",data=df1)

plt.title("Number of Property Sold in Melbourne",size=15);
# Number of property Unsold



unsold=home_data[home_data["Method"]!="S"]

unsold1=unsold[unsold["Method"]!="SP"]

unsold2=unsold1[unsold1["Method"]!="SA"]



# Count plot



plt.figure(figsize=(8,6))

sns.set(style="darkgrid")

sns.countplot(x="Method",data=unsold2)

plt.title("Number of Property Unsold in Melbourne",size=15);
# Distribution of Price against the Method



fig,ax=plt.subplots(figsize=(12,6))

sns.violinplot(y="Price",x="Method",data=home_data,ax=ax);

plt.title("Distribution of Price Vs Property Sold or Unsold",size=19);
# Distribution of price over the last 10 year across all the method



fig,ax=plt.subplots(figsize=(14,8))

sns.violinplot(x="YearBuilt",y="Price",hue="Method",data=year_data,ax=ax);

plt.title("Distribution of Price Over Last 10 years Vs Method ", size=16);
# Relationship between Price and distnace From CBD

dist=home_data[home_data["Distance"]>0]

plt.figure(figsize=(12,8))

sns.scatterplot(x="Distance",y="Price",data=dist)

plt.title("Distance From CBD and Price Anlaysis",size=16);
# Price analysis based on distance from CBD and regionname

fig= px.scatter(home_data,x="Distance",y="Price",color="Regionname",title="Price Vs Distance")

fig.show()
sns.lmplot(x="Distance",y="Price",data=home_data, x_estimator=np.mean)

plt.title("Price Vs Distance",size=14);
# Distribution of Price Vs CouncilArea



fig,ax=plt.subplots(figsize=(14,9))

ax=sns.boxplot(y="CouncilArea",x="Price",data=home_data,whis=np.inf)

ax.set_title("CouncilArea Vs Price Distribution",size=18);
# CouncilArea and Region



home_data.groupby(["Regionname","CouncilArea"])["Price"].mean().reset_index()
## BuildingArea Vs Price



sns.lmplot(x="BuildingArea",y="Price",hue="Regionname",data=home_data)

plt.title("Building Area and Price Analysis",size=14);
# Landsize Vs Price Analysis

sns.lmplot(x="Landsize",y="Price",hue="Type",data=home_data)

plt.title("Price Vs Landsize",size=15);
# Landsize Vs Price across different region



sns.scatterplot(x="Landsize",y="Price",hue="Regionname",data=home_data)

plt.title("Price Vs Landsize",size=15);
# Analysis of Price Vs  selected features

data=home_data[["Rooms","Distance","Bathroom","Landsize","YearBuilt","Car","Date","Propertycount"]]

price=home_data["Price"]

fig=plt.figure(figsize=(15,20))

for i in range(len(data.columns)):

    fig.add_subplot(3,3,i+1)

    sns.scatterplot(x=data.iloc[:,i],y=price)

plt.tight_layout()

plt.show()
# Prepare the data for correlation



corr=home_data.corr()



# Relation between different variable

fig,ax=plt.subplots(figsize=(14,9))

sns.heatmap(corr,annot=True,cmap = 'coolwarm',linewidth = 1,annot_kws={"size": 11})

plt.title("Correlation Among Different Variable",size=15);
common=home_data.groupby(['Regionname','Type','Rooms','Bathroom'])['Price'].count().reset_index()

SM=common[common["Regionname"]=="Southern Metropolitan"].sort_values(by="Price",ascending=False)
NM=common[common["Regionname"]=="Northern Metropolitan"].sort_values(by="Price",ascending=False)

NM.head(10)
# Common type of house, rooms and number of bathroom in Eastern Region



EM=common[common["Regionname"]=="Eastern Metropolitan"].sort_values(by="Price",ascending=False)

EM.head(10)
# Reasonable Price for 2 Bedroom unit 



two_rooms=home_data.groupby(["Regionname","Type","Rooms","Bathroom"]).Price.median().reset_index()

two_rooms[two_rooms["Rooms"]==2].sort_values(by="Price").reset_index(drop=True)
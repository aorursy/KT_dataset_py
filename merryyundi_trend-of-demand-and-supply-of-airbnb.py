# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly
from plotly.offline import iplot,init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Read the files into dataframe
listings = pd.read_csv("../input/listings.csv")
reviews = pd.read_csv("../input/reviews.csv")
calendar = pd.read_csv("../input/calendar.csv")

# Any results you write to the current directory are saved as output.
# Print info of listings
print(listings.info())
# Preview of listings
listings.head()
reviews.sample(10)
# "Host_Since": converted from "host_since"
# https://chrisalbon.com/python/basics/strings_to_datetime/
from datetime import datetime, timedelta
listings["Host_Since"] = pd.to_datetime(listings["host_since"])

# "Listing_Age"
# https://stackoverflow.com/questions/42521107/python-pandas-find-number-of-years-between-two-dates
listings["today"] = pd.to_datetime("2019-02-11")
listings["Hosting_Age"] = np.round(((listings["today"]- listings["Host_Since"])/timedelta(days=365)),2)

# "Host_Since_Year": Extracted from "Host_Since"
# https://stackoverflow.com/questions/30405413/python-pandas-extract-year-from-datetime-dfyear-dfdate-year-is-not/33757291
listings["Host_Since_Year"] = listings["Host_Since"].dt.year

# "Host_Since_Year_Month": Extracted from "Host_Since"
# https://datatofish.com/concatenate-values-python/
listings["Host_Since_Year_Month"] = (listings["Host_Since"].dt.year).map(str) + "-" + (listings["Host_Since"].dt.month).map(str)
listings["Host_Since_Year_Month"] = pd.to_datetime(listings["Host_Since_Year_Month"])
# Description of "id"
print("Description of id\n",listings["id"].describe())

print("-"*25)
# Calculate number of unique listings
print("Number of unique listings:",listings["id"].nunique())
# Calculate unique number of listings group by "neighbourhood_cleansed"
distric_listings = listings["id"].groupby(listings["neighbourhood_cleansed"]).nunique()
dis_listings = pd.DataFrame(distric_listings)

# Convert index of a pandas dataframe into a column 
# https://stackoverflow.com/questions/20461165/how-to-convert-index-of-a-pandas-dataframe-into-a-column
dis_listings.reset_index(level=0, inplace=True) 
dis_listings
# Barchart of unique listings in each district
trace = go.Bar(
                x = dis_listings.neighbourhood_cleansed,
                y = dis_listings.id,
                name = "Num of listings",
                marker = dict(color = 'rgba(255, 14, 5, 1.0)'),
                text = dis_listings.neighbourhood_cleansed)

layout = go.Layout(xaxis= dict(zeroline=False),
                   yaxis= dict(title= 'Num of listings'),
                   barmode="group",# barmode="relative"
                   title = 'Number of Airbnb Beijing listings by district') 

# Convert trace(plotly.graph_objs._bar.Bar) to a list 
data = [trace] 
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Print the frequency of multi-listings
(listings["calculated_host_listings_count"]>1).value_counts()
print("Number of unique hosts:",listings["host_id"].nunique())
# Calculate number of unique hosts in different districts
district_hosts = listings["host_id"].groupby(listings["neighbourhood_cleansed"]).nunique()
dis_hosts = pd.DataFrame(district_hosts)
dis_hosts.reset_index(level=0,inplace=True)
dis_hosts.head()
# Plot bar chart of unique hosts by district
trace = go.Bar(x=dis_hosts.neighbourhood_cleansed,
               y=dis_hosts.host_id,
               marker=dict(color="rgba(50,255,255,1.0)"),
               text=dis_hosts.neighbourhood_cleansed)
        
layout = go.Layout(xaxis= dict(zeroline=False),
                   yaxis= dict(title= 'Num of hosts',zeroline=False),
                   barmode="group",# barmode="relative"
                   title = 'Number of Airbnb Beijing hosts by district') 
                
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Calculate the number of listings each host have
host_listing = pd.DataFrame(listings["calculated_host_listings_count"].groupby(listings["host_id"]).count())
host_listing = host_listing.reset_index()
host_listing.sort_values(by="calculated_host_listings_count",ascending=False).head(10)
# Generate new feature host_group representing which group the host is in according to the listings they own.
# https://datatofish.com/if-condition-in-pandas-dataframe/
# To combine filtering conditions in Pandas, use bitwise operators ('&' and '|') not pure Python ones ('and' and 'or')
host_listing.loc[host_listing.calculated_host_listings_count<2,"host_group"]="1"
host_listing.loc[(host_listing.calculated_host_listings_count>=2) & (host_listing.calculated_host_listings_count<5) ,"host_group"]="2-4"
host_listing.loc[(host_listing.calculated_host_listings_count>=5) & (host_listing.calculated_host_listings_count<10) ,"host_group"]="5-9"
host_listing.loc[(host_listing.calculated_host_listings_count>=10) & (host_listing.calculated_host_listings_count<20) ,"host_group"]="10-19"
host_listing.loc[(host_listing.calculated_host_listings_count>=20) & (host_listing.calculated_host_listings_count<30) ,"host_group"]="20-29"
host_listing.loc[(host_listing.calculated_host_listings_count>=30) & (host_listing.calculated_host_listings_count<40) ,"host_group"]="30-39"
host_listing.loc[(host_listing.calculated_host_listings_count>=40) & (host_listing.calculated_host_listings_count<50) ,"host_group"]="40-49"
host_listing.loc[(host_listing.calculated_host_listings_count>=50) & (host_listing.calculated_host_listings_count<100) ,"host_group"]="50-99"
host_listing.loc[(host_listing.calculated_host_listings_count>=100) & (host_listing.calculated_host_listings_count<=222) ,"host_group"]="100-222"
group = pd.DataFrame(host_listing["host_group"].value_counts())
group = group.reset_index()

trace = go.Bar(x=group.host_group,
               y=group["index"],
               marker=dict(color="orange"),
               orientation="h",
               name="Num of hosts",
               text=group["index"])

layout = go.Layout(title="Number of hosts in each listing number group",
                   xaxis=dict(title="Number of hosts"),
                   yaxis=dict(title="Group"))

data=[trace]
fig = go.Figure(data=data,layout=layout)
        
iplot(fig)
(host_listing["calculated_host_listings_count"]>1).value_counts()
print("Number of superhosts",listings[listings["host_is_superhost"]=="t"]["host_id"].nunique())
print("Number of listings from superhosts",listings[listings["host_is_superhost"]=="t"]["id"].nunique())
super_listings=pd.DataFrame(listings[listings["host_is_superhost"]=="t"]["id"].groupby(listings.neighbourhood_cleansed).nunique())
super_listings=super_listings.reset_index()
nonsuper_listings=pd.DataFrame(listings[listings["host_is_superhost"]=="f"]["id"].groupby(listings.neighbourhood_cleansed).nunique())
nonsuper_listings=nonsuper_listings.reset_index()
trace1 = go.Bar(x=list(super_listings.neighbourhood_cleansed),
                y=list(super_listings.id),
                name="super listings")
trace2 = go.Bar(x=list(nonsuper_listings.neighbourhood_cleansed),
                y=list(nonsuper_listings.id),
                name="unsuper listings")

data = [trace1,trace2]

layout = go.Layout(xaxis = dict(zeroline=False),
                   yaxis = dict(title="Num of listings"),
                   barmode = "stack",
                   title = "Distribution of super/unsuper listings by district in Airbnb Beijing")
fig = go.Figure(data=data,layout=layout)
iplot(fig)
print(listings["room_type"].nunique())
print(listings["room_type"].value_counts())
# How do room type distribute in different districts?
dis_roomtype = listings["id"].groupby([listings["neighbourhood_cleansed"],listings["room_type"]]).nunique()
dis_roomtype = pd.DataFrame(dis_roomtype)
new_dis_roomtype = dis_roomtype.reset_index()
new_dis_roomtype
# Pie chart of room type
trace = go.Pie(values=listings["room_type"].value_counts(),
               labels=["Entire home/apt","Private room","Shared room"])
layout = go.Layout(title="Percentage of room type in Airbnb Beijing")
fig = go.Figure([trace],layout)
iplot(fig)
listings["accommodates"].describe()
# Plot a histogram of accommodates
trace = go.Histogram(x=listings["accommodates"],
                     marker=dict(color="#ff3399"))

layout = go.Layout(xaxis=dict(title="Accommodates"),
                   yaxis=dict(title="Num of accommodates"),
                   title="Histogram of accommodates")
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
a=listings[listings["room_type"]=='Entire home/apt']
b=listings[listings["room_type"]=='Private room']
# Plot a histogram of accommodates
trace = go.Histogram(x=a["accommodates"],
                     marker=dict(color="#ff3399"))

layout = go.Layout(xaxis=dict(title="Accommodates"),
                   yaxis=dict(title="Num of accommodates"),
                   title="Histogram of accommodates")
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
print(listings["bathrooms"].describe())
print(listings["bathrooms"].value_counts())
# Plot a histogram of bathrooms
trace = go.Histogram(x=listings["bathrooms"],
                     marker=dict(color="#ff9900"))

layout = go.Layout(xaxis=dict(title="Bathrooms"),
                   yaxis=dict(title="Num of bathrooms"),
                   title="Histogram of bathrooms")
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Divivde bathrooms into 2 groups, bathroom = 1 or bathroom >1
listings.loc[listings["bathrooms"]<1,"Bathroom"]="1-"
listings.loc[listings["bathrooms"]==1,"Bathroom"]="1"
listings.loc[listings["bathrooms"]>1,"Bathroom"]="1+"
listings["Bathroom"].value_counts()
listings["bedrooms"].describe()
listings["bedrooms"].value_counts()
# Plot a histogram of bathrooms
trace = go.Histogram(x=listings["bedrooms"],
                     marker=dict(color="#ff9900"))

layout = go.Layout(xaxis=dict(title="Bedrooms"),
                   yaxis=dict(title="Num of bedrooms"),
                   title="Histogram of bedrooms")
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
listings["beds"].describe()
listings["beds"].value_counts()
# Plot a histogram of beds
trace = go.Histogram(x=listings["beds"],
                     marker=dict(color="#ff9900"))

layout = go.Layout(xaxis=dict(title="Beds"),
                   yaxis=dict(title="Num of beds"),
                   title="Histogram of beds")
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
listings[["id","availability_30"]]
print("Description of availability",listings["availability_365"].describe())
print("Available more than half a year\n",(listings["availability_365"]>60).value_counts())
# Plot a histogram of availability_365
trace = go.Histogram(x=listings["bedrooms"],
                     marker=dict(color="#ff9900"))

layout = go.Layout(xaxis=dict(title="Bedrooms"),
                   yaxis=dict(title="Num of bedrooms"),
                   title="Histogram of bedrooms")
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
listings[["security_deposit"]].head(100)
listings[["amenities"]].head()
listings["number_of_reviews"].describe()

listings["Hosting_Age"].describe()
# Draw a histogram to show the distribution of listing age
trace = go.Histogram(x=listings["Hosting_Age"],
                     xbins=dict(start=0,end=9,size=0.5),
                     marker=dict(color="gold"))
layout = go.Layout(xaxis=dict(title="Hosting_Age"),
                   yaxis=dict(title="Frequency"),
                   title="Distribution of Listing Age in Airbnb Beijing")
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)

mean_age = pd.DataFrame(listings["Hosting_Age"].groupby(listings["neighbourhood_cleansed"]).mean())
mean_age = mean_age.reset_index()
# Plot bar chart of unique hosts by district
trace = go.Bar(x=mean_age["neighbourhood_cleansed"],
               y=mean_age["Hosting_Age"],
               marker=dict(color='pink'))
        
layout = go.Layout(xaxis= dict(zeroline=False),
                   yaxis= dict(title= 'Mean Age',zeroline=False),
                   barmode="group",# barmode="relative"
                   title = 'Mean Age of hosting by district') 
                
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
trace1 = go.Bar(x=new_dis_roomtype[new_dis_roomtype["room_type"]=="Entire home/apt"].neighbourhood_cleansed,
                y=new_dis_roomtype[new_dis_roomtype["room_type"]=="Entire home/apt"].id,
                name="Entire home/apt")
trace2 = go.Bar(x=new_dis_roomtype[new_dis_roomtype["room_type"]=="Private room"].neighbourhood_cleansed,
                y=new_dis_roomtype[new_dis_roomtype["room_type"]=="Private room"].id,
                name="Private room")
trace3 = go.Bar(x=new_dis_roomtype[new_dis_roomtype["room_type"]=="Shared room"].neighbourhood_cleansed,
                y=new_dis_roomtype[new_dis_roomtype["room_type"]=="Shared room"].id,
                name="Shared room")

data = [trace1,trace2,trace3]

layout = go.Layout(xaxis = dict(zeroline=False),
                   yaxis = dict(title="Num of listings"),
                   barmode = "stack",
                   title = "Distribution of room types by district in Airbnb Beijing")
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# "Price": price per night for number of included guests
# https://stackoverflow.com/questions/42192323/convert-pandas-dataframe-to-float-with-commas-and-negative-numbers
listings["Price"] = listings.price.str.strip('$') # Why the scale of price is USD?
listings["Price"] = listings.Price.str.replace(',','')
listings["Price"] = pd.to_numeric(listings.Price,errors='coerce')
listings["Price"].describe()

# Security_deposit: another continous value assiociated with the cost
listings["Security_deposit"] = listings.security_deposit.str.strip('$') 
listings["Security_deposit"] = listings.security_deposit.str.replace(',','')
listings["Security_deposit"] = pd.to_numeric(listings.Security_deposit,errors='coerce')
listings["Security_deposit"].describe()

# Cleaning_fee: additional cost at the top of rent
listings["Cleaning_fee"] = listings.cleaning_fee.str.strip('$') 
listings["Cleaning_fee"] = listings.cleaning_fee.str.replace(',','')
listings["Cleaning_fee"] = pd.to_numeric(listings.Cleaning_fee,errors='coerce')
listings["Cleaning_fee"].describe()

# Extra_people: cost of additional person per night
listings["Extra_people"] = listings.extra_people.str.strip('$') 
listings["Extra_people"] = listings.extra_people.str.replace(',','')
listings["Extra_people"] = pd.to_numeric(listings.Extra_people,errors='coerce')
listings["Extra_people"].describe()

# Extra_people: cost of additional person per night
listings["Extra_people"] = listings.extra_people.str.strip('$') 
listings["Extra_people"] = listings.extra_people.str.replace(',','')
listings["Extra_people"] = pd.to_numeric(listings.Extra_people,errors='coerce')
listings["Extra_people"].describe()
# When is the earliest and latest listing in Beijing Airbnb data?
print(listings["Host_Since"].min())
print(listings["Host_Since"].max())
# Count unique numbers of listings Groupby "Host_Since_Year"
listings_year = pd.DataFrame(listings["id"].groupby(listings["Host_Since_Year"]).nunique())
listings_year = listings_year.reset_index()

# Compute Cumulative Sum by date (year)
# https://stackoverflow.com/questions/42691405/cumulative-sum-by-date-month
listings_year["year_cumsum"] = listings_year["id"].cumsum()
# Draw a line to show the number of increased listings vs years since 2010
trace = go.Scatter(x=listings_year["Host_Since_Year"],
                   y=listings_year["id"],
                   mode = "lines+markers",
                   name="Num of listings",
                   marker = dict(color = "blue"),
                   text= listings_year.Host_Since_Year)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of listings",zeroline=False,),
                   title="Number of new listings in each year since 2010 in Airbnb Beijing")
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Calculate number of listings in each month period
listings_month = pd.DataFrame(listings["id"].groupby(listings["Host_Since_Year_Month"]).nunique())
listings_month = listings_month.reset_index()
listings_month["month_cumsum"] = listings_month["id"].cumsum()
# Draw a line to show the increased number of listings vs months since 2010-08
trace = go.Scatter(x=listings_month["Host_Since_Year_Month"],
                   y=listings_month["id"],
                   mode = "lines+markers",
                   name="Num of new listings",
                   marker = dict(color = 'purple'))

layout = go.Layout(xaxis=dict(title="Year-Month",zeroline=False),
                   yaxis=dict(title="Num of listings",zeroline=False,),
                   title="Number of new listings vs month in Airbnb Beijing")
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Draw a line to show the number of accumulative listings vs years since 2010
trace = go.Scatter(x=listings_year["Host_Since_Year"],
                   y=listings_year["year_cumsum"],
                   mode = "lines+markers",
                   name="Num of listings",
                   marker = dict(color = "rgba(255, 112, 2, 1.0)"),
                   text= listings_year.Host_Since_Year)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Accumulative sum of listings",zeroline=False,),
                   title="Accumulative number of listings year by year since 2010 in Airbnb Beijing")
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Draw 2 lines to show the number of new listings and totoal number of listings vs years since 2010
trace1 = go.Scatter(x=listings_year["Host_Since_Year"],
                   y=listings_year["id"],
                   mode = "lines+markers",
                   name="Num of new listings",
                   marker = dict(color = "blue"),
                   text= listings_year.Host_Since_Year)

trace2 = go.Scatter(x=listings_year["Host_Since_Year"],
                   y=listings_year["year_cumsum"],
                   mode = "lines+markers",
                   name="Num of total listings",
                   marker=dict(color="rgba(255, 112, 2, 1.0)"),
                   text= listings_year.Host_Since_Year)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of listings",zeroline=False,),
                   title="Number of new listings and total listings since 2010 in Airbnb Beijing")
data=[trace1,trace2]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Count unique numbers of listings Groupby "Host_Since_Year" and "neighbourhood_cleansed".
# https://chrisalbon.com/python/data_wrangling/pandas_apply_operations_to_groups/
listings_dis_year = pd.DataFrame(listings["id"].groupby([listings["neighbourhood_cleansed"],listings["Host_Since_Year"]]).nunique())
listings_dis_year = listings_dis_year.reset_index()
listings_dis_year
dis =  listings_dis_year["neighbourhood_cleansed"].unique()
colors = ["blue","red","yellow","green","pink","grey","black",
          "#ff99ff","#3399ff","#0099cc","#cc9900","#ff3399","#ff6600","#009900","#99cc00","#ff9900"]

def traces(a,df,column,colors):
    data = []
    for i in range(len(a)):
        trace = go.Scatter(x=df[df["neighbourhood_cleansed"]==a[i]].Host_Since_Year,
                           y=df[df["neighbourhood_cleansed"]==a[i]][column],
                           mode = "lines+markers",
                           name=a[i],
                           marker = dict(color = colors[i]))
        data.append(trace)
    return data

data =traces(dis,listings_dis_year,"id",colors)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of listings",zeroline=False,),
                   title="Number of new listings by Year and by Geo in Airbnb Beijing")

fig = go.Figure(data=data,layout=layout)
iplot(fig)
reviews.info()
print("Number of unique reviews:",reviews["id"].nunique())
print("Number of unique listings with:",reviews["listing_id"].nunique())
reviews[reviews["reviewer_name"].isnull()]
reviews[reviews["comments"].isnull()]
# "date": converted from str type "date" to datetime type "date"
# https://chrisalbon.com/python/basics/strings_to_datetime/
from datetime import datetime, timedelta
reviews["Date"] = pd.to_datetime(reviews["date"])
# "date_year": Extracted from "date"
# https://stackoverflow.com/questions/30405413/python-pandas-extract-year-from-datetime-dfyear-dfdate-year-is-not/33757291
reviews["date_year"] = reviews["Date"].dt.year

# "date_year_month": Extracted from "date"
# https://datatofish.com/concatenate-values-python/
reviews["date_year_month"] = (reviews["Date"].dt.year).map(str) + "-" + (reviews["Date"].dt.month).map(str)
reviews["date_year_month"] = pd.to_datetime(reviews["date_year_month"])
reviews.head()
# Count unique numbers of reviews Groupby "date_year"
review_year = pd.DataFrame(reviews["id"].groupby(reviews["date_year"]).nunique())
review_year = review_year.reset_index()

# Compute Cumulative Sum by date (year)
# https://stackoverflow.com/questions/42691405/cumulative-sum-by-date-month
review_year["year_cumsum"] = review_year["id"].cumsum()
review_year
# Draw a line to show the number of new reviews by year since 2010
trace = go.Scatter(x=review_year["date_year"],
                   y=review_year["id"],
                   mode = "lines+markers",
                   name="Num of reviews",
                   marker = dict(color = "#ff8c66"),
                   text= review_year.date_year)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of reviews",zeroline=False,),
                   title="Number of new reviews in each year since 2010 in Airbnb Beijing")
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Let's combine listing supply and demand together
trace1 = go.Scatter(x=review_year["date_year"],
                   y=review_year["id"],
                   mode = "lines+markers",
                   name="Num of new reviews",
                   marker = dict(color = "#ff8c66"),
                   text= review_year.date_year)

trace2 = go.Scatter(x=listings_year["Host_Since_Year"],
                   y=listings_year["id"],
                   mode = "lines+markers",
                   name="Num of new listings",
                   marker = dict(color = "blue"),
                   text= listings_year.Host_Since_Year)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of reviews",zeroline=False,),
                   title="Number of new reviews vs listings in each year since 2010 in Airbnb Beijing")

data=[trace1,trace2]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Draw a line to show the number of total reviews by year since 2010
trace = go.Scatter(x=review_year["date_year"],
                   y=review_year["year_cumsum"],
                   mode = "lines+markers",
                   name="Num of reviews",
                   marker = dict(color = "#cc33ff"),
                   text= review_year.date_year)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of reviews",zeroline=False,),
                   title="Number of total reviews until each year since 2010 in Airbnb Beijing")
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Let's combine total listing supply and demand together
trace1 = go.Scatter(x=review_year["date_year"],
                   y=review_year["year_cumsum"],
                   mode = "lines+markers",
                   name="Num of new reviews",
                   marker = dict(color = "#cc33ff"),
                   text= review_year.date_year)

trace2 = go.Scatter(x=listings_year["Host_Since_Year"],
                   y=listings_year["year_cumsum"],
                   mode = "lines+markers",
                   name="Num of new listings",
                   marker = dict(color = "blue"),
                   text= listings_year.Host_Since_Year)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of reviews",zeroline=False,),
                   title="Number of total reviews vs listings until each year since 2010 in Airbnb Beijing")

data=[trace1,trace2]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Break down into months
# Count unique numbers of reviews Groupby "date_year_month"
review_month = pd.DataFrame(reviews["id"].groupby(reviews["date_year_month"]).nunique())
review_month = review_month.reset_index()

# Compute Cumulative Sum by date (year)
# https://stackoverflow.com/questions/42691405/cumulative-sum-by-date-month
review_month["month_cumsum"] = review_month["id"].cumsum()
review_month
# Draw a line to show the number of new reviews by month since 2010
trace = go.Scatter(x=review_month["date_year_month"],
                   y=review_month["id"],
                   mode = "lines+markers",
                   name="Num of reviews",
                   marker = dict(color = "#33cc33"),
                   text= review_month.date_year_month)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of reviews",zeroline=False,),
                   title="Number of new reviews in each month since 2010 in Airbnb Beijing")
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Let's combine listing supply and demand together
trace1 = go.Scatter(x=review_month["date_year_month"],
                   y=review_month["id"],
                   mode = "lines+markers",
                   name="Num of new reviews",
                   marker = dict(color = "#ff8c66"),
                   text= review_month.date_year_month)

trace2 = go.Scatter(x=listings_month["Host_Since_Year_Month"],
                   y=listings_month["id"],
                   mode = "lines+markers",
                   name="Num of new listings",
                   marker = dict(color = "blue"),
                   text= listings_month.Host_Since_Year_Month)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of reviews",zeroline=False,),
                   title="Number of total reviews vs listings in each year since 2010 in Airbnb Beijing")

data=[trace1,trace2]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Draw a line to show the number of total reviews by month since 2010
trace = go.Scatter(x=review_month["date_year_month"],
                   y=review_month["id"],
                   mode = "lines+markers",
                   name="Num of reviews",
                   marker = dict(color = "#339933"),
                   text= review_month.date_year_month)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of reviews",zeroline=False,),
                   title="Number of total reviews until each month since 2010 in Airbnb Beijing")
data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Let's combine listing supply and demand together
trace1 = go.Scatter(x=review_month["date_year_month"],
                    y=review_month["month_cumsum"],
                    mode = "lines+markers",
                    name="Num of total reviews",
                    marker = dict(color = "#ff8c66"),
                    text= review_month.date_year_month)

trace2 = go.Scatter(x=listings_month["Host_Since_Year_Month"],
                    y=listings_month["month_cumsum"],
                    mode = "lines+markers",
                    name="Num of total listings",
                    marker = dict(color = "blue"),
                    text= listings_month.Host_Since_Year_Month)

layout = go.Layout(xaxis=dict(title="Year",zeroline=False),
                   yaxis=dict(title="Num of reviews",zeroline=False,),
                   title="Number of total reviews vs listings in each year since 2010 in Airbnb Beijing")

data=[trace1,trace2]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
print("Description of Price\n",listings["Price"].describe())
print(listings["Price"].isnull().sum())
# Plot a histogram of price
trace = go.Histogram(x=listings["Price"],
                     marker=dict(color="green"))

layout = go.Layout(xaxis=dict(title="Price"),
                   yaxis=dict(title="Num of Listings"),
                   title="Histogram of Price")
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Let calculate the Natural logarithmic value of price
trace = go.Histogram(x=np.log(listings['Price']),
                     marker=dict(color="green"))
layout = go.Layout(xaxis=dict(title="Log_Price"),
                   yaxis=dict(title="Num of Listings"),
                   title="Histogram of Log Price")
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
print("Median price of each district:",listings.Price.groupby(listings.neighbourhood_cleansed).median())
# Draw a box plot showing the distribution of room price in each district
trace = go.Box(x=listings.neighbourhood_cleansed,
               y=listings.Price,
               name="Price",
               marker = dict(color = 'rgb(122, 12, 240)'))

layout = go.Layout(xaxis=dict(zeroline=False),
                   yaxis=dict(title="Price(¥)"),
                   title="Box chart of room price by district in Airbnb Beijing")

data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Draw a box plot showing the distribution of room price in each type
trace = go.Box(x=listings.room_type,
               y=listings.Price,
               name="Price",
               marker = dict(color = 'rgba(3, 255, 240,1.0)'))

layout = go.Layout(xaxis=dict(zeroline=False),
                   yaxis=dict(title="Price(¥)"),
                   title="Box chart of room price by room type in Airbnb Beijing")

data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
price = listings[["id","neighbourhood_cleansed","host_is_superhost",
                  "room_type","accommodates","bathrooms","bedrooms","beds",
                  "review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location",
                  "review_scores_value",
                  "number_of_reviews","Price"]]
price.head()
import plotly.figure_factory as ff

price_score = price.corr()
figure = ff.create_annotated_heatmap(
    z=price_score.values,
    x=list(price_score.columns),
    y=list(price_score.index),
    annotation_text=price_score.round(2).values,
    showscale=True)
iplot(figure)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

price_origin = pd.get_dummies(price)

price_origin = price_origin.dropna(axis=0,)

X = price_origin.drop(['Price'],axis=1, inplace=False)   
X = np.array(X).astype(np.float)

X_scaled = scale(X)

y = price_origin[['Price']]  
y = np.array(y).astype(np.float)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.3, random_state=0)

knn = KNeighborsRegressor(algorithm='brute')

knn.fit(X_train, y_train)

y_predicted = model.predict(X_test)

df = listings[["calculated_host_listings_count","Price"]]
df.loc[listings["calculated_host_listings_count"]>1,"group"] = "Multi-listing"
df.loc[listings["calculated_host_listings_count"]==1,"group"] = "Mono-listing"
# Draw a box plot showing the distribution of room price in Multi-listing group and mono-listing group
trace = go.Box(x=df.group,
               y=listings.Price,
               name="Price",
               marker = dict(color = 'rgba(100, 15, 24,1.0)'))

layout = go.Layout(xaxis=dict(zeroline=False),
                   yaxis=dict(title="Price(¥)"),
                   title="Box chart of room price by listing type in Airbnb Beijing")

data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
print("Number of listings over 10K",(listings["Price"]>10000).value_counts())
print("Number of listings under 50",(listings["Price"]<50).value_counts())
Taregt: Price
listings["accommodates"].value_counts()
# Generate a "group" feature to divide number of reviews for each listing.
lis_reviews.loc[lis_reviews["id"]<5,"Group"]="1-4"
lis_reviews.loc[(lis_reviews["id"]>=5) & (lis_reviews["id"]<10),"Group"]="5-9"
lis_reviews.loc[(lis_reviews["id"]>=10) & (lis_reviews["id"]<20),"Group"]="10-19"
lis_reviews.loc[(lis_reviews["id"]>=20) & (lis_reviews["id"]<30),"Group"]="20-29"
lis_reviews.loc[(lis_reviews["id"]>=30) & (lis_reviews["id"]<40),"Group"]="30-39"
lis_reviews.loc[(lis_reviews["id"]>=40) & (lis_reviews["id"]<50),"Group"]="40-49"
lis_reviews.loc[(lis_reviews["id"]>=50) & (lis_reviews["id"]<100),"Group"]="50-99"
lis_reviews.loc[(lis_reviews["id"]>=100) & (lis_reviews["id"]<=312),"Group"]="100-312"
group_review = pd.DataFrame(lis_reviews["Group"].value_counts(sort=False))
group_review = group_review.reset_index()
trace = go.Bar(x=group_review["index"],
               y=group_review["Group"],
               name="Number of listings",
               text=group_review["index"],
               marker=dict(color="#ff0066"))
layout = go.Layout(title="Number of listings in each review group",
                   xaxis=dict(title="Review group"),
                   yaxis=dict(title="Number of listings"))
fig = go.Figure([trace],layout)
iplot(fig)
# Generate 2 new features: review year and review year_month 
reviews["date"] = pd.to_datetime(reviews["date"])
reviews["year"] = reviews["date"].dt.year
reviews["year_month"] = pd.to_datetime((reviews["date"].dt.year).map(str) + '-' + (reviews["date"].dt.month).map(str))
reviews.head()
year_review = pd.DataFrame(reviews["id"].groupby(reviews["year"]).nunique())
year_review = year_review.reset_index()
year_review["year_sum"] = year_review["id"].cumsum()
year_review
month_review = pd.DataFrame(reviews["id"].groupby(reviews["year_month"]).nunique())
month_review = month_review.reset_index()
month_review["month_sum"] = month_review["id"].cumsum()
month_review.head()
# Draw a line to show the number of reviews in each year 
trace = go.Scatter(x=year_review["year"],
                    y=year_review["id"],
                    name="Num of reviews in each year",
                    marker=dict(color="red"))


layout = go.Layout(title="Number of reviews in each year and month",
                   xaxis=dict(title="Year"),
                   yaxis=dict(title="Num of reviews"))
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
#  Draw a line to show the number of reviews in each month 
trace = go.Scatter(x=month_review["year_month"],
                    y=month_review["id"],
                    name="Num of reviews in each month",
                    marker=dict(color="blue"))

layout = go.Layout(title="Number of reviews in each year and month",
                   xaxis=dict(title="Year"),
                   yaxis=dict(title="Num of reviews"))
data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
dis_review = pd.DataFrame(listings["number_of_reviews"].groupby(listings["neighbourhood_cleansed"]).sum())
dis_review = dis_review.reset_index()
dis_review
# Draw a bar chart to show the distribution of reviews in different districts
trace = go.Bar(x=dis_review.neighbourhood_cleansed,
               y=dis_review.number_of_reviews,
               name="Number of reviews",
               marker=dict(color="#ff9900"),
               text=dis_review.neighbourhood_cleansed)

layout = go.Layout(title="Number of reviews in each district",
                   yaxis=dict(title="Number of reviews"))
data=[trace]
fig=go.Figure(data=data,layout=layout)
iplot(fig)
type_review = pd.DataFrame(listings["number_of_reviews"].groupby(listings["room_type"]).sum())
type_review = type_review.reset_index()
type_review
# Pie chart of room type
trace = go.Pie(values=type_review["number_of_reviews"],
               labels=type_review["room_type"])

layout = go.Layout(title="Percentage reviews in each room type")
fig = go.Figure([trace],layout)
iplot(fig)
reviews["comments"][10]
print("Number of listings without review score:",listings["review_scores_rating"].isnull().sum())
print(listings["review_scores_rating"].describe())
print((listings["review_scores_rating"]).value_counts())

scores = listings[["review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location","review_scores_value","host_total_listings_count","Price","Hosting_Age"]]
scores.head()
import plotly.figure_factory as ff

corr_scores = scores.corr()
figure = ff.create_annotated_heatmap(
    z=corr_scores.values,
    x=list(corr_scores.columns),
    y=list(corr_scores.index),
    annotation_text=corr_scores.round(2).values,
    showscale=True)
iplot(figure)
score_group=listings[["review_scores_rating"]]

# Generate a new feature "Group"
# https://stackoverflow.com/questions/5124376/convert-nan-value-to-zero 
score_group.loc[np.isnan(score_group.review_scores_rating),"Group"]="None"
score_group.loc[(score_group.review_scores_rating >=1) & (score_group.review_scores_rating <50),"Group"]="1-50"
score_group.loc[(score_group.review_scores_rating >=50) & (score_group.review_scores_rating <60),"Group"]="50-59"
score_group.loc[(score_group.review_scores_rating >=60) & (score_group.review_scores_rating <70),"Group"]="60-69"
score_group.loc[(score_group.review_scores_rating >=70) & (score_group.review_scores_rating <80),"Group"]="70-79"
score_group.loc[(score_group.review_scores_rating >=80) & (score_group.review_scores_rating <90),"Group"]="80-89"
score_group.loc[(score_group.review_scores_rating >=90) & (score_group.review_scores_rating <95),"Group"]="90-94"
score_group.loc[(score_group.review_scores_rating >=95) & (score_group.review_scores_rating <100),"Group"]="95-99"
score_group.loc[score_group.review_scores_rating==100,"Group"]="100"
df3 = pd.DataFrame(score_group.Group.value_counts())
df3 = df3.reset_index()
df3
# Draw a bar chart to show the distribution of "review_scores_rating"

trace = go.Bar(x=df3["index"],
               y=df3.Group,
               name="Number of listings",
               marker=dict(color="#00ccff"),
               text=df3["index"])

layout= go.Layout(title="Number of listings in different score group")

data=[trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Draw a box plot showing the distribution of room price in each district
trace = go.Box(x=listings.neighbourhood_cleansed,
               y=listings.review_scores_rating,
               name="Review Score",
               marker = dict(color = "#ff3399"))

layout = go.Layout(xaxis=dict(zeroline=False),
                   yaxis=dict(title="Review Score"),
                   title="Box chart of Review Score by district in Airbnb Beijing")

data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Draw a box plot showing the distribution of room price in each district
trace = go.Scatter(x=listings.number_of_reviews,
                   y=listings.review_scores_rating,
                   name="Review Score",
                   mode="markers",
                   marker = dict(color = "#ff3399"))

layout = go.Layout(xaxis=dict(title="Number of Reviews"),
                   yaxis=dict(title="Review Score"),
                   title="Scatter of Review Score vs Number of Reviews")

data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
# Draw a box plot showing the distribution of room price in each district
trace = go.Scatter(x=listings.Price,
                   y=listings.review_scores_rating,
                   name="Price",
                   mode="markers",
                   marker = dict(color="#339933"))

layout = go.Layout(xaxis=dict(zeroline=False),
                   yaxis=dict(title="Price"),
                   title="Scatter of Review Score vs Price")

data = [trace]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
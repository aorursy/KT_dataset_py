# load libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns
# read csv data files in dataframes

calendar = pd.read_csv('../input/boston/calendar.csv')

listings = pd.read_csv("../input/boston/listings.csv")

reviews = pd.read_csv("../input/boston/reviews.csv")
calendar.head()
calendar.available.value_counts()
# check percentages of missing values in each column

calendar.isnull().mean()
# check first five rows in listings

listings.head()
# view column names 

listings.columns
# check statistics of numerical columns 

listings.describe() 
# check top 5 rows of reviews 

reviews.head()
# check missing values 

reviews.isnull().mean()
# check date range

print("The earliest date in the data set is {}; the latest date is {}"

      .format(reviews.date.min(),reviews.date.max()))
# convert date from string to datetime object



calendar["date"] = pd.to_datetime(calendar["date"])

calendar.date.describe()
# calendar is coded as f and t string objects

calendar.available.value_counts()
# convert f to 0 (not available), and t to 1 (available)

calendar["available"]=calendar["available"].apply(lambda x:1 if x=="t" else 0)

calendar.available.value_counts()
# compute availability (percentage of units available) of to each day

# store in calendar_daily dataframe

#calendar_daily = calendar.groupby("date")["available"].mean().reset_index()



calendar_daily = calendar.groupby("date")["available"].agg(["sum","mean"]).reset_index()



calendar_daily.rename(columns={"sum":"available_count","mean":"availability"},inplace=True)                                         



calendar_daily.head()
# view overall trend of availability by day

sns.set_style("darkgrid")

plt.figure(figsize=(12,4))



ax=sns.lineplot(x=calendar_daily.date,y=calendar_daily.availability)

ax.set(xlabel="Date",ylabel="Availability", title="How does availability change by day?")



# set y tick values to % 

y_tickvalue=['{:,.0f}'.format(x*100) + '%' for x in ax.get_yticks()]

ax.set_yticklabels(y_tickvalue)



# set x tick increments 

months = mdates.MonthLocator()

ax.xaxis.set_major_locator(months)



plt.savefig("availability_trend_daily.png")
# check trend of available unit counts 



# view overall trend of availability by day

sns.set_style("darkgrid")

plt.figure(figsize=(16,6))



ax=sns.lineplot(x=calendar_daily.date,y=calendar_daily.available_count)

ax.set(xlabel="Date",ylabel="Total Available Units", title="How does total available count change by day?")



# set x tick increments 

months = mdates.MonthLocator()

ax.xaxis.set_major_locator(months)



plt.savefig("available_counts_trend_daily.png")

# explore trend throughout the week

calendar["day_of_week"] = calendar["date"].dt.dayofweek

calendar_weekly = calendar.groupby("day_of_week")["available"].agg(["sum","mean"]).reset_index()

calendar_weekly.rename(columns={"sum":"available_count","mean":"availability"},inplace=True)

calendar_weekly
plt.figure(figsize=(10,8))

sns.set_style("darkgrid")

ax=sns.lineplot(x=calendar_weekly.day_of_week,y=calendar_weekly.availability)

ax.set(xlabel="Day of Week",ylabel="Availability", title="How does availability change throughout the week?")

ax.set_xticks([0,1,2,3,4,5,6])

ax.set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])



# set y tick values to % 

y_tickvalue=['{:,.1f}'.format(x*100) + '%' for x in ax.get_yticks()]

ax.set_yticklabels(y_tickvalue)

plt.savefig("availability_weekly.png")
# Check total available units

plt.figure(figsize=(10,8))

sns.set_style("darkgrid")

ax=sns.lineplot(x=calendar_weekly.day_of_week,y=calendar_weekly.available_count)

ax.set(xlabel="Day of Week",ylabel="Total Available Listings", title="Total Avaiable Units throughout the Week")

ax.set_xticks([0,1,2,3,4,5,6])

ax.set_xticklabels(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])





plt.savefig("available_count_weekly.png")
# listing price is coded as a string

listings.price.head()
# convert listing price from string to numeric

listings["price"]=listings["price"].replace('\$|\,','',regex=True).astype(float)

listings["price"].describe()
# check neighbourhood_cleansed

listings.neighbourhood_cleansed.value_counts()
neighbourhood_list=listings["neighbourhood_cleansed"].dropna().unique()

sorted_neighbourhood=sorted(neighbourhood_list)

sorted_neighbourhood
# check room_type variable

listings.room_type.value_counts()
# create clustered bar plot for average price by Neighbourhood and room type



g=sns.FacetGrid(listings,row="room_type",hue="room_type",height=5,aspect=2,margin_titles=True)

g.map(sns.barplot,"price","neighbourhood_cleansed",ci=None,order=sorted_neighbourhood).fig.subplots_adjust(wspace=.05, hspace=.15)

g.add_legend(title="Room Type")

g.set(xlabel="Price",ylabel="Neighbourhood",title="Average Nightly Rental Rate by Neighbourhood and Room Type")

plt.savefig("average_price_by_neighbourhood_room_type.png",bbox_inches='tight')
# check number of rows and columns

listings.shape
listings.bedrooms.value_counts()
listings.bathrooms.value_counts()
listings.beds.value_counts()
listings.review_scores_rating.describe()
# select feature variables, and drop rows with one or more NAs 



listings_rm_nan = listings.dropna(how="any",subset=["neighbourhood_cleansed","bedrooms","bathrooms",

                                                   "room_type","review_scores_rating","beds"],axis=0)

y=listings_rm_nan["price"]



# create dummy codes for categorical values 

X_categorical = listings_rm_nan[["room_type","neighbourhood_cleansed"]]

X_categorical_dummy_coded = pd.get_dummies(X_categorical)



# check how many rows left after removing nans, check the number of columns 

X_categorical_dummy_coded.shape
X_num = listings_rm_nan[["bedrooms","bathrooms","review_scores_rating","beds"]]



#concatenate columns of numerical and categorical features 

X=pd.concat([X_num,X_categorical_dummy_coded],axis=1)



# check to make sure X and y have equal number of rows

X.shape[0]==y.shape[0]
# split X, y to test and train 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)





linear_model = LinearRegression(normalize=True)

linear_model.fit(X_train,y_train)

linear_model.score(X_test,y_test)

y_test_predicted = linear_model.predict(X_test)
# plot y_test_predict and y_test

plt.figure(figsize=(6,6))

ax=sns.scatterplot(y_test,y_test_predicted)

sns.lineplot(x=y_test,y=y_test,ax=ax,color="r",alpha=0.5, label="100% Perfect Fit")

ax.set(xlabel="Actual Rate",ylabel="Predicted Rate", title="Predicted Rate v.s. Actual Rate")

plt.savefig("predicted_rate_vs_actual_rate.png",bbox_inches='tight')
import statsmodels.api as sm

results=sm.OLS(y_train,X_train).fit() #ordinary least squares 

results.summary()
import pandas as pd
df = pd.read_csv("../input/superstore-simple-assessment/Assessment Python_ Dataset_superstore_simple.csv")

df.head()
#1. which order and customer id that has the most sales?



#the order that has the most sales



df.loc[df["sales"] == df["sales"].max()]
#to locate customer id that has the most sales, I need to sum every order from each customer, and find the maximum value

#here i make a new dataframe so that i dont get confused lmao



dfnew1 = df.groupby(by=["customer_id"]).sum()

dfnew12 = dfnew1.reset_index()

dfnew12
#the customer id that has the most sales



dfnew12.loc[dfnew12["sales"] == dfnew12["sales"].max()]
#2. What are sub-categories in the 'Office Supplies' product category? and how much is the total profit for each sub-category?



df.groupby(by=["category", "sub_category"]).sum()
#3. How many orders result in losses?



#losses means negative profit

df.loc[df["profit"] < 0]
#There are 1869 orders that result loss
#4.Among these 3 customer id, JE-16165, KH-16510, AD-10180, which one has the most total sales: ?



#locate the three customer id orders

dfnew4 = df.loc[df["customer_id"].isin(["JE-16165","KH-16510","AD-10180"])]

dfnew4.head(10)
#sum up the sales



dfnew4.groupby(by=["customer_id"]).sum()
#5. Create a new dataframe named "yearly_sales" contains total sales, number of customers, and total profit for each year.

#Which year has the most profit?



#I need to check the data type because I need to extract the year data



df = pd.read_csv("../input/superstore-simple-assessment/Assessment Python_ Dataset_superstore_simple.csv")

print(df.dtypes)
#I change the data type of order date to datetime



df["order_date"]= pd.to_datetime(df["order_date"])

print(df.dtypes)
#now that it is changed, I want to add it into a new column



df['year'] = pd.DatetimeIndex(df["order_date"]).year

df.head()
#sum up the data



dfnew5 = df.groupby(by=["year"]).sum()

dfnew5
#total sales and profit each year



dfnew51 = dfnew5[["sales","profit"]]

dfnew51
#I create another dataframe to count the number of customers



dfnew52 = df[["year","customer_id"]]

dfnew52.head()
#some customer made two or more orders, I need to avoid duplicate customer id



dfnew52.drop_duplicates()
#number of customer each year



dfnew53 = dfnew52.groupby(by=["year"]).count()

dfnew53
#merge dataframes into one



pd.merge(dfnew51, dfnew53,

        how= "left", on= "year")
#6. Make a scatter plot between sales and profit for 2014 and 2015. Differentiate the colors between each year.



df.head()
#delete data for 2016 and 2017.



year2016 = df[df["year"] == 2016].index

df.drop(year2016 , inplace=True)

year2017 = df[df["year"] == 2017].index

df.drop(year2017 , inplace=True)



df.head()
import plotly



import plotly.graph_objects as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True)
#create the scatter plot



data6 = go.Scatter(

    x=df.sales,

    y=df.profit,

    mode="markers",

    name="Sales vs Profit"

)



fig = go.Figure(data=data6)

iplot(fig)
import numpy as np
#differentiate the color marker



df["marker_color"] = np.where(

    df.year == 2014,

    "green",

    "blue"

)

df.head()
data6 = go.Scatter(

    x = df.sales,

    y = df.profit,

    mode = "markers",

    marker = {

        "color": df.marker_color,

    }

)



fig = go.Figure(data=data6)

iplot(fig)



#scatter plot between sales and profit for year 2014 and 2015
#6. Create a barchart for 10 customer that has the most total sales in 2015



#I create a new dataframe for 2015 data only

dfnew7 = df.loc[df["year"] == 2015]

dfnew7
#sum up the sales



dfnew7.groupby(by=["customer_id"]).sum()
#locate 10 customer that has the most sales



dfnew71 = dfnew7.groupby(by=["customer_id"]).sum()

dfnew72 = dfnew71.sort_values(by=["sales"], ascending=False)

dfnew72.head(10)
#I need to reset the index to create the chart



dfnew73 = dfnew72.reset_index()

dfnew74 = dfnew73.head(10)

dfnew74
#create the barchart



data7 = go.Bar(

    x = dfnew74.customer_id,

    y = dfnew74.sales

)

fig = go.Figure(data=data7)

iplot(fig)
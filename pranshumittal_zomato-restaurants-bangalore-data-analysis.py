# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import plotly.express as px
import math
import re
import warnings
#loading the data set
restaurannt_data = pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns", 200)

main_df = pd.DataFrame(restaurannt_data)
main_df.head(100)
main_df.info()
#Droping irrelevant columns
main_df.drop(columns = ["url","phone"], inplace = True)
#bringing the names column to the front
main_df.insert(loc = 0, column = "Name", value = main_df["name"])
#dropping the old name column
main_df.drop(columns = "name", inplace = True)
main_df.info()
#Modify the ratings column
main_df.rename(columns = {"rate" :"ratings"}, inplace = True)
main_df["ratings"] = main_df["ratings"].replace(to_replace = ["/5"], value = '', regex = True).str.strip()
#Lets check the unique values of ratings column
main_df["ratings"].unique()
#Replace all non digit values to 0
main_df["ratings"].replace(to_replace = ["-","NEW",np.NaN], value = 0, inplace = True)
main_df.ratings.unique()
#Finally convert the data type of the ratings column to float
main_df["ratings"] = main_df["ratings"].astype('float')

main_df.ratings.dtype
main_df["menu_item"].replace(to_replace = "[]", value = np.NaN, inplace = True)
main_df["reviews_list"].replace(to_replace = "[]", value = np.NaN, inplace = True)
main_df.isna().sum()
#We will drop the rows where value of approx cost and rest_type is not mentioned
main_df.dropna(subset = ["rest_type","approx_cost(for two people)","cuisines"], inplace = True)
main_df.isna().sum()
#Now we will drop the menu_item and dish liked columns as approximately 50% of the values in these columns are missing
#Also we will drop the reviews list column which we forgot to drop earlier as the column is of no use to us in our analysis
main_df.drop(["menu_item","dish_liked","reviews_list"],axis = 1, inplace = True)

#We have handled the missing values in the data frame
main_df.isna().sum()
#Now we will convert the approx cost column values to integer data type
main_df["approx_cost(for two people)"] = main_df["approx_cost(for two people)"].replace(to_replace = "[,]", value = "", regex = True)
main_df["approx_cost(for two people)"] = main_df["approx_cost(for two people)"].astype("int")
#Funtion to categorize the cost for two values
def cost_for_two(value):
    if value < 200:
        return "<200"
    elif value < 500:
        return "200-500"
    elif value < 800:
        return "500-800"
    elif value < 1500:
        return "800-1500"
    elif value <3000:
        return "1500-3000"
    else:
        return ">3000"
#Categorizing the value of the "approx cost for two people"
main_df["cost_for_two"] = main_df["approx_cost(for two people)"].apply(lambda x:cost_for_two(x))
#We will remove all rows with zero ratings and store then in another data frame for future separate analysis

new_rest_df = main_df[main_df["ratings"] == 0]
res_df = main_df[main_df["ratings"] != 0]
#Shape of two data frames we created
res_df.shape, new_rest_df.shape, main_df.shape
main_df["listed_in(city)"].unique()
main_df["online_order"].unique(), main_df["book_table"].unique()
online_orders = ((main_df["online_order"].value_counts()/main_df["online_order"].count())*100).round(2)
print(online_orders)
sn.barplot(x = online_orders.index, y = online_orders)
plt.title("Percentage of outlets accepting online orders", fontsize = 16)
plt.ylabel("Percentage %", fontsize = 14)
plt.xticks(fontsize = 14)
plt.show()
book_table = (main_df['book_table'].value_counts() * 100 /main_df['book_table'].count()).round(2)
print(book_table)
sn.barplot(x = book_table.index, y = book_table)
plt.title("Percentage of outlets with Table booking service", fontsize = 16)
plt.ylabel("Percentage %", fontsize = 14)
plt.xticks(fontsize = 14)
plt.show()
print(res_df["ratings"].describe())
plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
sn.distplot(res_df.ratings, hist = False)
plt.subplot(1,2,2)
sn.boxplot(res_df["ratings"])
plt.show()
potential_outliers_count = res_df[res_df["ratings"]<2.5]["ratings"].count()
print(f"Total number of potential outliers with ratings less than 2.5 are {potential_outliers_count}")
top_rated = res_df[res_df.sort_values(by = ["listed_in(type)","ratings"], ascending = [True, False])["ratings"]>4.0]
all_outlets = res_df.sort_values(by = ["listed_in(type)"])
all_outlets_count = all_outlets["listed_in(type)"].value_counts()
top_rated_count = top_rated["listed_in(type)"].value_counts()
perc_by_each_type = (top_rated_count*100/all_outlets_count).round(2)
perc_by_each_type.sort_values(ascending = False, inplace = True)
perc_by_each_type
print(all_outlets_count)
sn.barplot(y = perc_by_each_type.index, x = perc_by_each_type,color = "blue", alpha = 0.5)
plt.title("Percentage of Top (more than 4) rated outlets in respective categories ", fontsize = 16)
plt.xlabel("Outlet Types", fontsize = 14)
plt.ylabel("Percentage", fontsize = 14)
plt.show()
number_of_outlets = main_df["listed_in(city)"].value_counts()
plt.figure(figsize = (16,10))
sn.barplot(y = number_of_outlets.index, x = number_of_outlets, color = "red", alpha = 0.5)
plt.title("Total number of outlets in each location", fontsize = 18)
plt.xlabel("Locations in city", fontsize = 15)
plt.ylabel("Number of Outlets", fontsize = 15)
plt.show()
fig = px.treemap(res_df, path = ["listed_in(city)","listed_in(type)"], 
                 color = "ratings",
                 height = 800,
                 title = "Plot showing Number of outlets in each category for each location ")
fig.show()
#Plotting a distribution graph for the cost for two column

plt.figure(figsize = (18,14))
temp_df = main_df["approx_cost(for two people)"].value_counts().sort_index()
plt.subplot(2,2,1)
sn.distplot(temp_df)
plt.xticks(rotation = 30 , fontsize = 12)
plt.xlabel("Cost", fontsize = 14)
plt.title("Cost for two - Distribution", fontsize = 16)

#Plotting a bar plot for the cost for two column 

temp_df2 = main_df["cost_for_two"].value_counts()
plt.subplot(2,2,2)
sn.barplot(x = temp_df2.index, y = temp_df2, color = "lightseagreen")
plt.xticks(rotation = 30 , fontsize = 12)
plt.xlabel("Cost", fontsize = 14)
plt.ylabel("Number of Outlets", fontsize = 14)
plt.title("Cost for two - Number of Outlets", fontsize = 16)

plt.subplot(2,2,3)
sn.boxplot(main_df["approx_cost(for two people)"])
plt.xlabel("Cost", fontsize = 14)
plt.title("Cost for two", fontsize = 16)
plt.show()

print("The Median cost for Meal for two in Bangalore is ", main_df["approx_cost(for two people)"].median())
fig = px.box(main_df, x = "listed_in(type)", y = "approx_cost(for two people)")
fig.update_layout(title = dict(text = "Cost for two for different types of outlets",
                                font = dict(size = 16, color = "black"),
                              x = 0.5),
                 xaxis = dict(title = "Type of Outlet"),
                 yaxis = dict(title = "Cost for Two"))
fig.show()
fig = px.box(main_df, x = "listed_in(city)", y = "approx_cost(for two people)")
fig.update_layout(title = dict(text = "Cost for two people in different locations",
                              font = dict(size = 16, color = "black"),
                              x = 0.5),
                 xaxis = dict(title = "Locations"),
                 yaxis = dict(title = 'Cost for two people'))
fig.add_shape( # add a horizontal "target" line
    type="line", line_color="purple", line_width=2, opacity=0.5, line_dash="dot",
    x0=0, x1=1, xref="paper", y0=3000, y1=3000, yref="y")
fig.show()
fig = px.histogram(main_df, x = "listed_in(type)",color = "online_order", 
                   log_y = False, color_discrete_sequence= ["darkcyan",'lightseagreen'])
fig.update_layout(title = dict(text = "Distribution of outlets accepting online orders",
                  x = 0.5,
                  font = dict(size = 16)),
                  yaxis = dict(title = "Count",type = "log", nticks = 3),
                 xaxis = dict(title = 'Outlet Types'))
fig.show()
fig = px.histogram(main_df, x = "listed_in(type)", color = "book_table", log_y= True, 
                   color_discrete_sequence=["indigo","darkorchid"])
fig.update_layout(title = dict(text = "Outlets offering Table booking Serive",
                              font = dict(size = 16),
                              x = 0.5),
                               yaxis = dict(title = "Count",
                                           type = "log",
                                           nticks = 3),
                               xaxis = dict(title = "Type of Outlet"))
fig.show()
fig = px.box(res_df, x = "listed_in(type)", y = 'ratings', color_discrete_sequence=["darkcyan"])
fig.update_layout(title = dict(text = "Analysis of different types of outlets based on Ratings",
                              x = 0.5,
                              font = dict(size = 18)),
                 xaxis = dict(title = "Type of Outlet"),
                 yaxis = dict(title = "Rating"))
fig.show()
fig = px.box(res_df , x = "cost_for_two", y = "ratings")
fig.update_layout(title = dict(text = "Analysis of Cost for two VS Ratings",
                              x = 0.5,
                              font = dict(size = 18)),
                 xaxis = dict(title = "Cost for Two"),
                 yaxis = dict(title = "Rating"))
fig.show()
df = res_df.groupby("cost_for_two")["ratings"].median()
df.sort_values(ascending = True, inplace = True)
df = pd.DataFrame(df)
df["Rank"] = [2,1,3,4,5,6]
df
sn.regplot(df["Rank"], df["ratings"])
from scipy import stats
stats.spearmanr(df)
cuisine = " "
for i in main_df["cuisines"]:
    for j in i.split(","):
        cuisine += j + ","
    cuisine += " "  
cuisine = list(set(cuisine.split(",")))
cuisine = [i.strip() for i in cuisine]
cuisine.remove("")
print("Total number of different cuisines is",len(cuisine))
df = {}
for i in cuisine:
    df[i] = len(main_df[main_df["cuisines"].str.contains(i)==True]["cuisines"])
df = pd.DataFrame(data = [list(df.keys()), list(df.values())], index = ["cuisine_type","num_of_outlets"]).T
df.sort_values("num_of_outlets", ascending = False, inplace = True)
df
fig = px.bar(df, "cuisine_type", "num_of_outlets", log_y= True)
fig.update_layout(title = dict(text = "Popularity of Different Cuisines",
                             font = dict(size = 18),
                             x = 0.5),
                xaxis = dict(title = "Cuisine Type"),
                yaxis = dict(title = "Number of Outlets", type = "log", nticks = 5))
fig.show()
famous_chains = main_df.groupby("listed_in(type)")
famous_chains = famous_chains.apply( lambda x: x["Name"].value_counts()).reset_index(drop = False)
famous_chains.rename(columns = {"level_1" : "name", "Name" : "total_outlets"}, inplace = True)
famous_chains = famous_chains.groupby("listed_in(type)").head(5)
px.treemap(famous_chains, path = ["listed_in(type)","name"], color = "total_outlets")
fig = px.histogram(new_rest_df,"listed_in(type)",color = "cost_for_two", 
                   log_y = True,
                  color_discrete_sequence= ["darkcyan","lightseagreen", "darkturquoise", "cadetblue","mediumturquoise"])
fig.update_layout(title = dict(text = "New Outlets Opened in different Categories",
                              font = dict(size = 18),
                              x = 0.5),
                 yaxis = dict(title = "Number of Outlets",
                             type = "log",
                             nticks = 5),
                 xaxis = dict(title = "Outlet Type"))
fig.show()
famous = res_df.sort_values(["listed_in(type)","votes"], ascending = [True,False]).groupby("listed_in(type)")

#Removing all the multiple outlets of different chains
famous = famous.apply(lambda x: x.drop_duplicates(subset = "Name")).reset_index(drop = True)
famous.groupby("listed_in(type)").head(5)[["Name","ratings","votes","listed_in(type)","listed_in(city)"]]
famous = famous.groupby("listed_in(type)")
famous_df = famous.head(5)[["Name","ratings","votes","listed_in(type)","listed_in(city)"]].sort_values("listed_in(city)")
famous_df
px.treemap(famous_df, path = ["listed_in(city)","listed_in(type)","Name"], hover_data = ["ratings","Name"], color = "votes")

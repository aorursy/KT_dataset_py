# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Important imports for the analysis of the dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Create the dataframe and check the first 8 rows

app_df = pd.read_csv("/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv")

app_df.head(8)
# Dropping columns that I will not use for this analysis

app_df_cut = app_df.drop(columns=['URL', 'Subtitle', 'Icon URL'])
app_df_cut.info()
# Most reviewed app

#app_df_cut.iloc[app_df_cut["User Rating Count"].idxmax()]



# A better way of seeing the most reviwed apps 

app_df_cut = app_df_cut.sort_values(by="User Rating Count", ascending=False)

app_df_cut.head(5)
# Get the columns "User Rating Count" and "Average User Rating" where they are both equal to NaN and set the

# values to 0.

app_df_cut.loc[(app_df_cut["User Rating Count"].isnull()) | (app_df_cut["Average User Rating"].isnull()),

               ["Average User Rating", "User Rating Count"]] = 0
# Check if there are any other missing values in those columns

app_df_cut.loc[(app_df_cut["User Rating Count"].isnull()) | (app_df_cut["Average User Rating"].isnull())]
# Get the column "In-app Purchases" where the value is NaN and set it to zero

app_df_cut.loc[app_df_cut["In-app Purchases"].isnull(),

               "In-app Purchases"] = 0
# Check if there are any NaN value in the "In-app Purchases" column

app_df_cut.loc[app_df_cut["In-app Purchases"].isnull()]
# Check if there are missing or 0 ID's

app_df_cut.loc[(app_df_cut["ID"] == 0) | (app_df_cut["ID"].isnull()),

              "ID"]
# Check for duplicates in the ID column

len(app_df_cut["ID"]) - len(app_df_cut["ID"].unique())



# The number of unique values is lower than the total amount of ID's, therefore there are duplicates among them.
# Drop every duplicate ID row

app_df_cut.drop_duplicates(subset="ID", inplace=True)

app_df_cut.shape
# Check if there are null values in the Size column

app_df_cut[(app_df_cut["Size"].isnull()) | (app_df_cut['Size'] == 0)]
# Drop the only row in which the game has no size

app_df_cut.drop([16782], axis=0, inplace=True)
# Convert the size to MB

app_df_cut["Size"] = round(app_df_cut["Size"]/1000000)

app_df_cut.head(5)
# Drop the row with NaN values in the "Price" column

app_df_cut = app_df_cut.drop(app_df_cut.loc[app_df_cut["Price"].isnull()].index)
# Check if there are any null values on the price column

app_df_cut.loc[app_df_cut["Price"].isnull()]
# Drop the rows with NaN values in the "Languages" column

app_df_cut = app_df_cut.drop(app_df_cut.loc[app_df_cut["Languages"].isnull()].index)
# Check if there are any null values on the "Languages" column

app_df_cut.loc[app_df_cut["Languages"].isnull()]
app_df_cut.info()
app_df_cut.to_csv("app_df_clean.csv", index=False)
app_df_clean = pd.read_csv("app_df_clean.csv")

app_df_clean.head()
# Transform the the string dates into datetime objects

app_df_clean["Original Release Date"] = pd.to_datetime(app_df_clean["Original Release Date"])

app_df_clean["Current Version Release Date"] = pd.to_datetime(app_df_clean["Current Version Release Date"])
app_df_clean.info()
# Make the figure

plt.figure(figsize=(16,10))



# Variables

years = app_df_clean["Original Release Date"].apply(lambda date: date.year)

size = app_df_clean["Size"]



# Plot a swarmplot

palette = sns.color_palette("muted")

size = sns.swarmplot(x=years, y=size, palette=palette)

size.set_ylabel("Size (in MB)", fontsize=16)

size.set_xlabel("Original Release Date", fontsize=16)

size.set_title("Time Evolution of the Apps' Sizes", fontsize=20)

plt.show()
# Make the figure

plt.figure(figsize=(16,10))



# Plot a countplot

palette1 = sns.color_palette("inferno_r")

apps_per_year = sns.countplot(x=years, data=app_df_clean, palette=palette1)

apps_per_year.set_xlabel("Year of Release", fontsize=16)

apps_per_year.set_ylabel("Amount", fontsize=16)

apps_per_year.set_title("Quantity of Apps per Year", fontsize=20)



# Write the height of each bar on top of them

for p in apps_per_year.patches:

    apps_per_year.annotate("{}".format(p.get_height()),

                          (p.get_x() + p.get_width() / 2, p.get_height() + 40),

                          va="center", ha="center", fontsize=16)
#Make a list of years from 2014 to 2018

years_lst = [year for year in range(2014,2019)]



#For loop to get a picture of the amount of games produced from August to December

for year in years_lst:

    from_August = app_df_clean["Original Release Date"].apply(lambda date: (date.year == year) & (date.month >= 8)).sum()

    total = app_df_clean["Original Release Date"].apply(lambda date: date.year == year).sum()

    print("In {year}, {percentage}% games were produced from August to December."

          .format(year=year,

                  percentage=round((from_August/total)*100, 1)))
# Make the figure

plt.figure(figsize=(16,10))



# Variables

price = app_df_clean["Price"]



# Plot a Countplot

palette2 = sns.light_palette("green", reverse=True)

price_vis = sns.countplot(x=price, palette=palette2)

price_vis.set_xlabel("Price (in US dollars)", fontsize=16)

price_vis.set_xticklabels(price_vis.get_xticklabels(), fontsize=12, rotation=45)

price_vis.set_ylabel("Amount", fontsize=16)

price_vis.set_title("Quantity of Each App per Price", fontsize=20)



# Write the height of the bars on top

for p in price_vis.patches:

    price_vis.annotate("{:.0f}".format(p.get_height()), # Text that will appear on the screen

                       (p.get_x() + p.get_width() / 2 + 0.1, p.get_height()), # (x, y) has to be a tuple

                       ha='center', va='center', fontsize=14, color='black', xytext=(0, 10), # Customizations

                       textcoords='offset points')
# Make the figure

plt.figure(figsize=(16,10))



# Variables

in_app_purchases = app_df_clean["In-app Purchases"].str.split(",").apply(lambda lst: len(lst))



# Plot a stripplot

palette3 = sns.color_palette("BuGn_r", 23)

in_app_purchases_vis = sns.stripplot(x=price, y=in_app_purchases, palette=palette3)

in_app_purchases_vis.set_xlabel("Game Price (in US dollars)", fontsize=16)

in_app_purchases_vis.set_xticklabels(in_app_purchases_vis.get_xticklabels(), fontsize=12, rotation=45)

in_app_purchases_vis.set_ylabel("In-app Purchases Available", fontsize=16)

in_app_purchases_vis.set_title("Quantity of In-app Purchases per Game Price", fontsize=20)

plt.show()
# Plot a distribution of the top 200 apps by their price



# Make the figure

plt.figure(figsize=(16,10))



# Plot a Countplot

palette4 = sns.color_palette("BuPu_r")

top_prices = sns.countplot(app_df_clean.iloc[:200]["Price"], palette=palette4)

top_prices.set_xlabel("Price (in US dollars)", fontsize=16)

top_prices.set_xticklabels(top_prices.get_xticklabels(), fontsize=12)

top_prices.set_ylabel("Amount", fontsize=16)

top_prices.set_title("Quantity of Each App per Price", fontsize=20)



# Write the height of the bars on top

for p in top_prices.patches:

    top_prices.annotate("{:.0f}".format(p.get_height()), 

                        (p.get_x() + p.get_width() / 2., p.get_height()),

                        ha='center', va='center', fontsize=14, color='black', xytext=(0, 8),

                        textcoords='offset points')
# Create the DataFrames needed

paid = app_df_clean[app_df_clean["Price"] > 0]

total_paid = len(paid)

free = app_df_clean[app_df_clean["Price"] == 0]

total_free = len(free)



# Make the figure and the axes (1 row, 2 columns)

fig, axes = plt.subplots(1, 2, figsize=(16,10))



# Free apps countplot

free_vis = sns.countplot(x="Average User Rating", data=free, ax=axes[0])

free_vis.set_xlabel("Average User Rating", fontsize=16)

free_vis.set_ylabel("Amount", fontsize=16)

free_vis.set_title("Free Apps", fontsize=20)



# Display the percentages on top of the bars

for p in free_vis.patches:

     free_vis.annotate("{:.1f}%".format(100 * (p.get_height()/total_free)),

                       (p.get_x() + p.get_width() / 2 + 0.1, p.get_height()),

                        ha='center', va='center', fontsize=14, color='black', xytext=(0, 8),

                        textcoords='offset points')

    

# Paid apps countplot

paid_vis = sns.countplot(x="Average User Rating", data=paid, ax=axes[1])

paid_vis.set_xlabel("Average User Rating", fontsize=16)

paid_vis.set_ylabel(" ", fontsize=16)

paid_vis.set_title("Paid Apps", fontsize=20)



# Display the percentages on top of the bars

for p in paid_vis.patches:

    paid_vis.annotate("{:.1f}%".format(100 * (p.get_height()/total_paid)),

                      (p.get_x() + p.get_width() / 2 + 0.1, p.get_height()),

                       ha='center', va='center', fontsize=14, color='black', xytext=(0, 8),

                       textcoords='offset points')
# Make the figure

plt.figure(figsize=(16,10))



# Make a countplot

palette5 = sns.color_palette("BuGn_r")

age_vis = sns.countplot(x=app_df_clean["Age Rating"], order=["4+", "9+", "12+", "17+"], palette=palette5)

age_vis.set_xlabel("Age Rating", fontsize=16)

age_vis.set_ylabel("Amount", fontsize=16)

age_vis.set_title("Amount of Games per Age Restriction", fontsize=20)



# Write the height of the bars on top

for p in age_vis.patches:

    age_vis.annotate("{:.0f}".format(p.get_height()), 

                        (p.get_x() + p.get_width() / 2., p.get_height()),

                        ha='center', va='center', fontsize=14, color='black', xytext=(0, 8),

                        textcoords='offset points')
# Create a new column that contains the amount of languages that app has available

app_df_clean["numLang"] = app_df_clean["Languages"].apply(lambda x: len(x.split(",")))
#Make the figure

plt.figure(figsize=(16,10))



#Variables

lang = app_df_clean.loc[app_df_clean["numLang"] <= 25, "numLang"]



#Plot a countplot

palette6 = sns.color_palette("PuBuGn_r")

numLang_vis = sns.countplot(x=lang, data=app_df_clean, palette=palette6)

numLang_vis.set_xlabel("Quantity of Languages", fontsize=16)

numLang_vis.set_ylabel("Amount of Games", fontsize=16)

numLang_vis.set_title("Quantity of Languages Available per Game", fontsize=20)



# Write the height of the bars on top

for p in numLang_vis.patches:

    numLang_vis.annotate("{:.0f}".format(p.get_height()), 

                        (p.get_x() + p.get_width() / 2. + .1, p.get_height()),

                        ha='center', va='center', fontsize=12, color='black', xytext=(0, 12),

                        textcoords='offset points')
#Amount of games that have only the English language

len(app_df_clean[(app_df_clean["numLang"] == 1) & (app_df_clean["Languages"] == "EN")])
#Amount of games that have only one language and is not English

len(app_df_clean[(app_df_clean["numLang"] == 1) & (app_df_clean["Languages"] != "EN")])
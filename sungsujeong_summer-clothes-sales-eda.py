# Import libraries



import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import missingno as msno



import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected = True)
# Import original data (csv file)

summer = pd.read_csv('../input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv')



# Print top five observations

summer.head()
# Data information

summer.info()
# Check null values, data type, and unique values in each column



null = summer.isnull().sum().to_frame(name='nulls').T

dtype = summer.dtypes.to_frame(name='dtypes').T

nunique = summer.nunique().to_frame(name='unique').T

pd.concat([null, dtype, nunique], axis=0)
# Statistic summary for continuous variables

summer.describe()
# Detect the total number of missing values in each column

summer_nullity = summer.isnull().sum()

print(summer_nullity)
# Visualize missing values 

# (1): as a matrix

msno.matrix(summer)

plt.show()



# (2): as a bar

msno.bar(summer)

plt.show()
# Missing patterns of dataset sorted by 'rating_#_count'

rating_sorted = summer.sort_values('rating_five_count')

msno.matrix(rating_sorted)

plt.show()
# Missing Patterns of the dataset sorted by 'has_urgency_banner'

urgency_sorted = summer.sort_values('has_urgency_banner')

msno.matrix(urgency_sorted)

plt.show()
# Visualize the correlation between the number of missing values in different columns 

# (1) as a heatmap 

msno.heatmap(summer)

plt.show()



#(2) as a dendrogram

msno.dendrogram(summer)

plt.show()
# Subset multiple columns

summer[["has_urgency_banner", "urgency_text"]]
# Count the values of the column has_urgency_banner

summer["has_urgency_banner"].value_counts()
# Count the values of the column urgency_text

summer["urgency_text"].value_counts()
# Check if the missing values in both columns match

print(summer.loc[summer["has_urgency_banner"].isna(), "urgency_text"].unique())



# Check if the non-missing values in both columns match

print(summer.loc[~summer["has_urgency_banner"].isna(), "urgency_text"].unique())
# Removing the column urgency_text

summer.drop(["urgency_text"], axis=1, inplace=True)



# Data imputation

# Re-defining the column has_urgency_banner and fill the missig values with 0

summer["has_urgency_banner"] = summer["has_urgency_banner"].fillna(0)



# Confirm changes made

summer["has_urgency_banner"]
# Group columns related to rating and define as rating_cols

rating_cols = ["rating", "rating_count", "rating_five_count", "rating_four_count", 

               "rating_three_count", "rating_two_count", "rating_one_count"]



# Define no_vote as the columns that have missing values

no_vote = summer.loc[summer[rating_cols].isna().any(axis=1), rating_cols]



# Set the value of 0 to each section that has missing values 

summer.loc[no_vote.index, rating_cols] = 0



# Print top five obs

summer.loc[no_vote.index, rating_cols].head()
# Remove unimportant variables or variables that have a few missing values

drop_cols = ["title", "currency_buyer", "tags", "product_color", "product_variation_size_id", 

             "product_variation_inventory", "shipping_option_name", "shipping_is_express", 

             "countries_shipped_to", "inventory_total", "origin_country", "merchant_title", 

             "merchant_name", "merchant_info_subtitle", "merchant_id", "merchant_profile_picture",

             "product_url", "product_picture", "product_id", "theme", "crawl_month"]



summer.drop(drop_cols, axis=1, inplace=True)

print(summer.columns)
# Remove duplicate rows - if any

summer.drop_duplicates(keep='first', inplace=True)
# Histogram for price



sns.set()

plt.figure(figsize=(8, 6))

plt.hist(summer['price'], bins=20, facecolor='navy', range=[0, 50])

plt.xlabel("price (€)", fontsize=15)

plt.ylabel("number of unit sold", fontsize=15)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title("Price Distribution", fontsize=17)

plt.show()
# Histogram for retail price



plt.subplot(1, 2, 1)

(summer['retail_price']).plot.hist(bins=50, figsize=(12, 6), facecolor='gray', range=[0, 175])

plt.xlabel("retail price (€)", fontsize=15)

plt.ylabel("number of unit sold", fontsize=15)

plt.title("Retail Price Distribution", fontsize=17)

plt.xticks(fontsize=13)

plt.xticks(fontsize=13)

plt.show()



plt.subplot(1, 2, 2)

np.log(summer['retail_price']+1).plot.hist(bins=50, figsize=(12, 6), facecolor='gray')

plt.xlabel("log(retail price + 1)", fontsize=15)

plt.ylabel("number of unit sold", fontsize=15)

plt.title("Retail Price Distribution", fontsize=17)

plt.xticks(fontsize=13)

plt.xticks(fontsize=13)

plt.show()
# Kernel Density Estimates (KDE) for both prices



plt.figure(figsize=(8, 6))

sns.kdeplot(summer['price'], shade=True)

sns.kdeplot(summer['retail_price'], shade=True)

plt.xlabel('price (€)', fontsize=15)

plt.ylabel('probability of density', fontsize=15)

plt.title("Price vs Retail Price", fontsize=17)

plt.show()
# Plot data using Empirical Cumulative Distribution Function (ECDF)

# Compute and define ECDF

def ecdf(data):

    n = len(data)

    x = np.sort(data)

    y = np.arange(1, n+1) / n

    return x, y



# Plot ECDFs and Comparison of ECDFs

x_pri, y_pri = ecdf(summer['price'])

x_re_pri, y_re_pri = ecdf(summer['retail_price'])



plt.figure(figsize=(8, 6))

plt.plot(x_pri, y_pri, marker='.', linestyle='none')

plt.plot(x_re_pri, y_re_pri, marker='.', linestyle='none', color='red')



plt.legend(('price', 'retail price'), loc='lower right')

plt.xlabel('price (€)', fontsize=15)

plt.ylabel('ECDF', fontsize=15)

plt.title("Price vs Retail Price (1)", fontsize=17)

plt.margins(0.02) # Keep data off plot edges

plt.show()



# Produce CDF using Seaborn

plt.figure(figsize=(8, 6))

sns.kdeplot(summer['price'], shade=True, cumulative=True)

sns.kdeplot(summer['retail_price'], shade=True, cumulative=True)

plt.xlabel('price (€)', fontsize=15)

plt.ylabel('CDF', fontsize=15)

plt.title("Price vs Retail Price (2)", fontsize=17)

plt.show()
# Check the distribution of shipping_option_price



plt.figure(figsize=(8, 6))

sns.violinplot('shipping_option_price', data=summer, palette='muted', 

               scale='count', inner='quartile')

plt.xlabel('shipping price range (€)', fontsize=15)

plt.title('Shipping Price Distribution', fontsize=17)

plt.show()
# Correlation between units_sold and price variables



plt.figure(figsize=(8, 6))

summer_price = summer[["units_sold", "price", "retail_price", "shipping_option_price"]]

sns.heatmap(summer_price.corr(), annot=True)

plt.title('Correlation between Units sold and Prices', fontsize=15)

plt.show()
# Count the number of usage of ad boots



plt.figure(figsize=(8, 6))

sns.countplot('uses_ad_boosts', data=summer, palette="Set2")

plt.xlabel('Ad-boots Usage', fontsize=15)

plt.ylabel('Count', fontsize=15)

plt.xticks([0, 1], ["No", "Yes"], fontsize=13)

plt.title('Ad-boots Usage', fontsize=17)

plt.show()
# Correlation between units_sold and uses_ad_boosts



corr = summer['uses_ad_boosts'].corr(summer['units_sold'])

print(f"The Correlation coefficient between units_sold and uses_ad_boosts is: {np.round(corr, 4)}")
# KDE plot for ad-boosts usage and unit sold



g = sns.FacetGrid(summer, hue='uses_ad_boosts', height=4, aspect=3)



g = (g.map(sns.kdeplot, 'units_sold', shade=True).add_legend())

max_units = summer['units_sold'].max()

g.set(xlim=(0, max_units))

plt.title('Ad-boosts Usage vs Unit Sold', fontsize=17)

plt.show()
# Correlation between unit sold and badges



plt.figure(figsize=(8, 6))

badge_group = summer[["badges_count", "badge_local_product", 

                      "badge_product_quality", "badge_fast_shipping", "units_sold"]]

sns.heatmap(badge_group.corr(), annot=True)

plt.title('Correlation between Units sold and Badges', fontsize=15)

plt.show()
# Association between units sold and badges on product



g = sns.catplot(x="badges_count", y="units_sold", data=summer, 

                kind='box', palette="Set3", aspect=2)

(g.set(ylim=(0, 60000))) 

plt.xlabel("total number of badges", fontsize=15)

plt.ylabel("unit sold", fontsize=15)

plt.title("Association between Unit sold and Badge", fontsize=17)

plt.show()
# Association between urgency banner and unit sold

# Use pointplot



sns.catplot(x="has_urgency_banner", y="units_sold", data=summer, kind="point", joint=True)

plt.xlabel("urgency banner", fontsize=15)

plt.ylabel("unit sold", fontsize=15)

plt.xticks([0.0, 1.0], ["No", "Yes"], fontsize=13)

plt.title("Association between has_urgency_banner and units_sold (1)", fontsize=15)

plt.show()
# Use violinplot



sns.catplot(x="has_urgency_banner", y='units_sold', data=summer, kind="violin", palette="Set1")

plt.xlabel("urgency banner", fontsize=15)

plt.ylabel("unit sold", fontsize=15)

plt.xticks([0.0, 1.0], ["No", "Yes"], fontsize=13)

plt.title("Association between has_urgency_banner and units_sold (2)", fontsize=15)

plt.show()
# Define function for range categorization of units_sold



def num_units_sold(units_sold):

    units_sold = int(units_sold)

    

    bracket = ''

    if units_sold in range(0, 100):

        bracket = '< 100'

    if units_sold in range(100, 1000):

        bracket = '100 - 1000'

    if units_sold in range(1000, 5000):

        bracket = '1000 - 5000'

    if units_sold in range(5000, 10000):

        bracket = '5000 - 10000'

    if units_sold in range(10000, 20000):

        bracket = '10000 - 20000'

    if units_sold in range(20000, 50000):

        bracket = '20000 - 50000'

    if units_sold in range(50000, 100000):

        bracket = '> 50000'

    return bracket
# Add up the all counts of each rating level in given range category



summer["num_units_sold"] = summer["units_sold"].apply(num_units_sold)

units_sold = summer['num_units_sold'].unique().tolist()

units_sold_groupby = summer.groupby("num_units_sold").agg({

    'rating_five_count': 'sum',

    'rating_four_count': 'sum',

    'rating_three_count': 'sum',

    'rating_two_count': 'sum',

    'rating_one_count': 'sum'})

units_sold_groupby.reset_index()

units_sold_groupby = units_sold_groupby.iloc[1:]

units_sold_groupby
# Check association between rating_#_count and units_sold using barchart



unit_range = ["< 100", "100 - 1000", "1000 - 5000", "5000 - 10000", 

              "10000 - 20000", "20000 - 50000", "> 50000"]

rat_1 = ["47.0", "2264.0", "13459.0", "21971.0", "32677.0", "48738.0", "17576.0"]

rat_2 = ["40.0", "1187.0", "7769.0", "13327.0", "21399.0", "33191.0", "12960.0"]

rat_3 = ["58.0", "2328.0", "16041.0", "27271.0", "46207.0", "70335.0", "27439.0"]

rat_4 = ["89.0", "3151.0", "21729.0", "36015.0", "65087.0", "93245.0", "35783.0"]

rat_5 = ["225.0", "7529.0", "54169.0", "90094.0", "164130.0", "226350.0", "88685.0"]



layout = go.Layout(barmode="stack")



fig = go.Figure(data=[

    go.Bar(x=unit_range, y=rat_5, name="rating_five_count"),

    go.Bar(x=unit_range, y=rat_4, name="rating_four_count"),

    go.Bar(x=unit_range, y=rat_3, name="rating_three_count"),

    go.Bar(x=unit_range, y=rat_2, name="rating_two_count"),

    go.Bar(x=unit_range, y=rat_1, name="rating_one_count")], layout=layout)

iplot(fig)
# define function for categorization of rating



def rating_category(rating):

    rating = int(rating)

    

    bracket = ''

    if rating in range (1, 3):

        bracket = "poor"

    if rating in range (3, 4):

        bracket = "fair"

    if rating in range (4, 6):

        bracket = "good"

    return bracket
# Correlation between unit sold and ratings using scatterplot



summer["avg_rating"] = summer["rating"].apply(rating_category)



plt.figure(figsize=(8, 6))



sns.scatterplot(x='units_sold', y='rating_count', hue='avg_rating', 

                palette='Pastel1', size='avg_rating', sizes=(10, 200), data=summer)

plt.xlabel("number of units sold", fontsize=15)

plt.ylabel("total number of rating", fontsize=15)

plt.xticks([0, 20000, 40000, 60000, 80000, 100000], ['0', '20k', '40k', '60k', '80k', '100k'], 

           fontsize=13)

plt.yticks([0, 5000, 10000, 20000], ['0', '5k', '10k', '20k'], fontsize=13)

plt.title("Correlation between unit sold and ratings", fontsize=17)

plt.show()
# Correlation coefficient between units_sold and merchant variables 



plt.figure(figsize=(8, 6))

merchant_group = summer[["merchant_rating_count", "merchant_rating", 

                      "merchant_has_profile_picture", "units_sold"]]

sns.heatmap(merchant_group.corr(), annot=True, cmap="YlGnBu")

plt.title('Correlation between units_sold and merchant variables', fontsize=15)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.show()
# Correlation between units_sold and merchant_profile & rating using lineplot



summer["merchant_profile"] = summer["merchant_has_profile_picture"].rename().apply(lambda x: "Yes" if x==1 else "No")



plt.figure(figsize=(8, 6))



sns.lineplot(x="units_sold", y="merchant_rating", hue="merchant_profile", hue_order=['Yes', 'No'], 

             style="merchant_profile", palette="ocean_r", data=summer)

plt.xlabel("number of units sold", fontsize=15)

plt.ylabel("mean of merchant rating", fontsize=15)

plt.xticks([0, 20000, 40000, 60000, 80000, 100000], ['0', '20k', '40k', '60k', '80k', '100k'], 

           fontsize=13)

plt.title("Correlation between units_sold and merchant_profile & rating", fontsize=17)

plt.show()
# Define function for categorization of merchant_rating



def avg_mer_rating(merchant_rating):

    merchant_rating = int(merchant_rating)

    

    bracket = ''

    if merchant_rating in range (0, 3):

        bracket = "poor"

    if merchant_rating in range (3, 4):

        bracket = "fair"

    if merchant_rating in range (4, 6):

        bracket = 'good'

    return bracket
# Correlation between units_sold and merchant rating



summer["avg_mer_rating"] = summer["merchant_rating"].apply(avg_mer_rating)

summer["merchant_profile"] = summer["merchant_has_profile_picture"].rename()



plt.figure(figsize=(8, 6))



sns.lineplot(x="units_sold", y="merchant_rating_count",  hue="avg_mer_rating", style="avg_mer_rating", 

             palette="magma_r", data=summer)

plt.xlabel("number of units sold", fontsize=15)

plt.ylabel("total number of merchant rating", fontsize=15)

plt.xticks([0, 20000, 40000, 60000, 80000, 100000], ['0', '20k', '40k', '60k', '80k', '100k'], 

           fontsize=13)

plt.yticks([0, 50000, 100000, 150000, 200000, 250000, 300000, 350000], 

           ['0', '50k', '100k', '150k', '200k', '250k', '300k', '350k'], fontsize=13)

plt.title('Correlation between units_sold and merchant_rating & count', fontsize=17)

plt.show()
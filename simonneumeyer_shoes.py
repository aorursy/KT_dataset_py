import pandas as pd

import os

import numpy

import matplotlib.pyplot as plot

import seaborn as sb

import random

import re

from datetime import datetime

import matplotlib.colors as mc

import colorsys

from scipy import interpolate
filepath_women = "../input/Datafiniti_Womens_Shoes.csv"

df = pd.read_csv(filepath_women)
print(df.shape)

df.head(10)
df.info()
df.describe()
categoricals = df.dtypes[df.dtypes == "object"].index.tolist()

df[categoricals].describe()
df[["colors", "prices.color"]].head(15)
df[["manufacturer", "manufacturerNumber"]].head(15)
columns_to_delete = ["asins", "colors", "dimension", "ean", "manufacturer", "manufacturerNumber", "weight", "prices.availability", "prices.condition", "prices.merchant", "prices.offer", "prices.returnPolicy", "prices.shipping"]

df.drop(columns_to_delete, axis = 1, inplace = True)
df.head(5)
print(df["prices.currency"].value_counts())

print(df["primaryCategories"].value_counts())
df.drop(["prices.currency", "primaryCategories"], axis = 1, inplace = True)
date_columns = ["dateAdded", "dateUpdated", "prices.dateAdded", "prices.dateSeen"]

df[date_columns].info()
df["prices.dateSeen"].value_counts()[:2]
df.drop(["prices.dateSeen"], axis = 1, inplace = True)

date_columns.remove("prices.dateSeen")
df_dates = df[date_columns].apply(func = lambda column : pd.to_datetime(column, errors = "ignore"), axis = 0)
#check number of missing values

print(df_dates["dateAdded"].isna().sum())

print(df_dates["dateUpdated"].isna().sum())

print(df_dates["prices.dateAdded"].isna().sum())
#replace missing values with median

timestamps = df_dates["prices.dateAdded"].map(na_action = "None", arg = lambda t : (t - numpy.datetime64('1970-01-01T00:00:00Z')) / numpy.timedelta64(1, 's'))

median = numpy.datetime64(datetime.utcfromtimestamp(timestamps.median()))

df_dates["prices.dateAdded"] = df_dates["prices.dateAdded"].fillna(median, axis = 0)
figure, ax = plot.subplots(figsize = (20,15))

figure.suptitle("Values of date columns", fontsize = 20)

plot.ylim(numpy.datetime64("2017-07-01"), numpy.datetime64("2018-05-01"))

plot.xlim(3000, 6000)

plot.scatter(x = range(len(df_dates.index)), y = df_dates["dateAdded"].values, c = "blue", s = 10, label = "dateAdded")

plot.scatter(x = range(len(df_dates.index)), y = df_dates["dateUpdated"].values, c = "green", s = 10, label = "dateUpdated")

plot.scatter(x = range(len(df_dates.index)), y = df_dates["prices.dateAdded"].values, c = "red", s = 10, label = "prices.dateAdded")

ax.legend()

x = plot.xticks([])
#check if any date column needs to be considered when grouping the data entries (by our interpretation of an product entity/ID)

non_date_columns = list(frozenset(df.columns) - frozenset(date_columns))

for column in df_dates.columns:

    print(column + ":")

    print(df.groupby(non_date_columns)[column].nunique().max())
df.drop(date_columns, axis = 1, inplace = True)

df = pd.concat([df, df_dates.dateUpdated], axis = 1)
def get_relationships(column):

    return df.groupby(column).apply(func = lambda frame : frame.apply(axis = 0, func = lambda col : col.nunique())).apply(

        axis = 0, func = lambda x : x.max()).values



relationship_matrix = pd.DataFrame([get_relationships(column) for column in df.columns], columns = df.columns, index = df.columns)
relationship_matrix
#upc (universal product code)

df[["id", "keys", "name", "prices.sourceURLs", "sourceURLs", "upc", "imageURLs"]].head()
df.drop(["keys", "name", "prices.sourceURLs", "sourceURLs", "upc", "imageURLs"], axis = 1, inplace = True)
df.rename(columns = {"prices.amountMax": "maxprice", "prices.amountMin": "minprice", "prices.color": "color", "prices.isSale": "sale",

                    "prices.size": "size", "dateUpdated": "date"}, inplace = True)
#convert string values to lower case

columns_nominal_categorical = ["brand", "categories", "color"]

for col in columns_nominal_categorical:

    df[col] = df[col].map(arg = lambda nominal : nominal.lower())
df.categories.value_counts().head(20)
def parse_category(category, words_categories, second_run = False, last_run = False):

    if(category.startswith("womens,shoes,")):

        words = category.split(",", 3)

        words_categories.append(words[2])

        return words[2]

    else:

        if second_run:

            hits = [cat for cat in words_categories if cat in category]

            if len(hits) > 0:

                return(hits[random.randint(0, len(hits) - 1)])

            else:

                return category

        else:

            if last_run and len(category.split(",")) >= 2:

                return "other"

            else:

                return category
random.seed(1)

words_categories = []

df_category = df["categories"].map(arg = lambda category : parse_category(category, words_categories))

df_category.value_counts()
words_categories = frozenset(words_categories)

words_categories = words_categories.union(frozenset(["work", "casual", "running", "dress"]))

df_category = df_category.map(arg = lambda category : parse_category(category, words_categories, second_run = True))

df_category.value_counts()
df_category = df_category.map(arg = lambda category : parse_category(category, words_categories, last_run = True))

df.categories = df_category
figure, ax = plot.subplots(figsize = (14,8))

figure.suptitle("Shoe categories", fontsize = 20)

plot.rcParams["ytick.labelsize"] = 15

plot.rcParams["xtick.labelsize"] = 12

p = df.categories.value_counts().head(10).plot.barh()

l = ax.set_xlabel("Number of products", fontsize = 15)
df.color.value_counts()[:15]
df.color.value_counts()[-15:]
figure, ax = plot.subplots(figsize = (14,8))

figure.suptitle("Shoe colors", fontsize = 20)

plot.rcParams["ytick.labelsize"] = 12

df.color.value_counts().head(10).plot.barh(color = "blue")

ax.set_ylabel("Color", fontsize = 15)

l = ax.set_xlabel("Number of products", fontsize = 15)
#This function weights the prices equally by id

def aggregate_price_by_columns(df, par_columns):

    price_min = df.groupby(par_columns + ["id"]).minprice.mean().reset_index().groupby(par_columns).minprice.mean()

    price_max = df.groupby(par_columns + ["id"]).maxprice.mean().reset_index().groupby(par_columns).maxprice.mean()

    distinct_products = df.groupby(par_columns).id.nunique()

    result = pd.concat([price_min, price_max, distinct_products], axis = 1)

    price_interval = result.apply(func = lambda row : row.maxprice - row.minprice, axis = 1)

    result = pd.concat([result, price_interval], axis = 1).rename(columns = {"id": "#products", 0 : "span",

                                                                            "minprice": "min_mean", "maxprice": "max_mean"})

    return result.sort_values(by = "min_mean", axis = 0)



def aggregate_price_by_id(df):

    price_min = df.groupby("id").minprice.mean()

    price_max = df.groupby("id").maxprice.mean()

    result = pd.concat([price_min, price_max], axis = 1)

    price_interval = result.apply(func = lambda row : row.maxprice - row.minprice, axis = 1)

    result = pd.concat([result, price_interval], axis = 1).rename(columns = {0 : "span", "minprice": "min_mean", "maxprice": "max_mean"})

    return result.sort_values(by = "min_mean", axis = 0)
prices_by_id = aggregate_price_by_id(df)

figure, ax = plot.subplots(figsize = (15,8))

figure.suptitle("Shoe prices", fontsize = 20)

plot.xlim(0, 175)

plot.rcParams["ytick.labelsize"] = 12

plot.rcParams["xtick.labelsize"] = 12

sb.kdeplot(prices_by_id.min_mean.values, color = "blue", label = "mean minimum price")

ax.set_xlabel("Price", fontsize = 15)

ax.set_ylabel("Density", fontsize = 15)

sb.kdeplot(prices_by_id.max_mean.values, color = "green", label = "mean maximum price")

l = ax.legend()
prices_by_brand = aggregate_price_by_columns(df, ["brand"])

prices_by_brand.min_mean[-5:]
#returns color shade for scalar input

def lighten_color(color, amount=0.5):

    #get hexadecimal value

    c = mc.cnames[color]

    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



#get max and mean number of products

mean_number_products = prices_by_brand["#products"].mean()

max_number_products = prices_by_brand["#products"].max()



#scale possible range for number of products on suitable range for input on func lighten_color

x = [1, mean_number_products, max_number_products]

y = [0.2, 1, 1.8]

f = interpolate.interp1d(x, y)



#calculate corresponding color for each brand's number of products

plot_color = [lighten_color("blue", f(number_products)) for number_products in prices_by_brand["#products"]]
figure, ax = plot.subplots(figsize = (20,10))

figure.suptitle("Prices by brand(1)", fontsize = 20)

plot.rcParams["ytick.labelsize"] = 12

plot.ylim(0, 220)

plot.bar(prices_by_brand.index.values, prices_by_brand.span, bottom = prices_by_brand.min_mean, color = plot_color)

ax.set_ylabel("Price", fontsize = 15)

ax.set_xlabel("Brands", fontsize = 15)

plot.scatter(x = prices_by_brand.index.values, y = prices_by_brand.min_mean, color = plot_color)

ticks = plot.xticks([])
prices_by_brand = prices_by_brand.sort_values(

    ascending = False, by = "#products", axis = 0).reset_index().loc[:9, :].set_index("brand").sort_values("min_mean", axis = 0)
figure, ax = plot.subplots(figsize = (20,10))

figure.suptitle("Prices by brand(2)", fontsize = 20)

plot.ylim(30, 90)

plot.bar(prices_by_brand.index.values, prices_by_brand.span, bottom = prices_by_brand.min_mean)

plot.rcParams["ytick.labelsize"] = 12

plot.rcParams["xtick.labelsize"] = 12

plot.scatter(x = prices_by_brand.index.values, y = prices_by_brand.min_mean, color = "blue")

l = ax.set_ylabel("Price", fontsize = 15)
prices_by_color = aggregate_price_by_columns(df, ["color"]).sort_values(

    ascending = False, by = "#products", axis = 0).reset_index().loc[:9, :].set_index("color").sort_values("min_mean", axis = 0)

prices_by_color
plot_color = ['navajowhite', 'navy', 'floralwhite', 'burlywood', 'gray', 'black', 'brown', 'blue', 'red', 'tan']
figure, ax = plot.subplots(figsize = (20,10))

figure.suptitle("Prices by color", fontsize = 20)

plot.ylim(40, 80)

plot.bar(prices_by_color.index.values, prices_by_color.span, bottom = prices_by_color.min_mean, color = plot_color)

plot.rcParams["ytick.labelsize"] = 12

plot.rcParams["xtick.labelsize"] = 12

l = ax.set_ylabel("Price", fontsize = 15)
prices_by_category = aggregate_price_by_columns(df, ["categories"]).sort_values(

    ascending = False, by = "#products", axis = 0).reset_index().loc[:9, :].set_index("categories").sort_values("min_mean", axis = 0)
figure, ax = plot.subplots(figsize = (20,10))

figure.suptitle("Prices by category", fontsize = 20)

plot.rcParams["ytick.labelsize"] = 12

plot.rcParams["xtick.labelsize"] = 15

plot.ylim(30, 140)

plot.bar(prices_by_category.index.values, prices_by_category.span, bottom = prices_by_category.min_mean)

l = ax.set_ylabel("Price", fontsize = 15)
df.sizes.value_counts()[:10]
print([size for size in df.sizes.values if not "," in size])
number_sizes = pd.DataFrame(df.sizes.map(arg = lambda size_list : len(size_list.split(",")))).rename(columns = {"sizes": "#sizes"})

price_span = pd.DataFrame(df.apply(func = lambda row : row.maxprice - row.minprice, axis = 1)).rename(columns = {0: "span"})

df_sizes = pd.concat([df, number_sizes, price_span], axis = 1)
df_sizes.corr()
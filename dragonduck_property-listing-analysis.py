import numpy as np

np.random.seed(101)

import requests

import time

import os

import requests

from bs4 import BeautifulSoup

import pandas as pd

import os

import re

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

import sklearn.model_selection

import sklearn.linear_model

import sklearn.feature_selection

import sklearn.preprocessing

import sklearn.metrics

import keras.models

import keras.layers
properties = pd.read_csv("../input/data_kaggle.csv")
properties.head()
properties = properties.loc[~properties["Price"].isna()]
incorrect_entries = np.sum(~properties["Price"].str.match(r"RM [0-9,]*$"))

print("There are {} entries in the wrong format.".format(incorrect_entries))
# Strip the price of the "RM" as well as commas

def strip_price(text):

    text = text.replace("RM", "")

    text = text.replace(",", "")

    text = text.strip()

    return int(text)

    

properties["Price"] = properties["Price"].apply(strip_price)
properties["Location"] = properties["Location"].str.lower()

properties["Location"] = properties["Location"].str.replace(r", kuala lumpur$", "")
sorted(properties["Location"].unique())
properties["Location"].value_counts().plot(logy=True);
significant_locations = properties["Location"].value_counts()[

    properties["Location"].value_counts() >= 100].index



properties = properties.loc[np.isin(properties["Location"], significant_locations)]
sorted(properties["Location"].unique())
sorted(properties["Rooms"].unique().astype(str))
def convert_room_num(rooms):

    try:

        if rooms.endswith("+"):

            return int(rooms[:-1])

        if re.search("[0-9]+\+[0-9]+", rooms) is not None:

            tmp = rooms.split("+")

            return int(tmp[0]) + int(tmp[1])

        if rooms == "20 Above":

            return 20

        if rooms == "Studio":

            return 1

        return int(rooms)

    except AttributeError:

        return rooms



properties["Rooms Num"] = properties["Rooms"].apply(convert_room_num)
properties["Rooms Num"].value_counts(dropna=False)
properties["Property Type"].value_counts()
def simplify_property_type(prop_type):

    super_types = [

        "Terrace/Link House", "Serviced Residence", "Condominium", 

        "Semi-detached House", "Bungalow", "Apartment", "Townhouse", 

        "Flat", "Residential Land", "Cluster House"]

    for super_type in super_types:

        if re.search(super_type, prop_type, flags=re.IGNORECASE) is not None:

            return super_type

    

    return prop_type



properties["Property Type Supergroup"] = properties["Property Type"].apply(simplify_property_type)
properties["Property Type Supergroup"].value_counts(dropna=False)
properties["Furnishing"].value_counts(dropna=False)
properties[["Size"]].sample(25)
def split_size(val, index=0):

    try:

        return val.split(":")[index].strip()

    except AttributeError:

        return val

    

properties["Size Type"] = properties["Size"].apply(split_size, index=0)

properties["Size Num"] = properties["Size"].apply(split_size, index=1)
properties["Size Type"].value_counts(dropna=False)
def convert_size_num(size):

    # Attempt to trim the numbers down. Most of this is done explicitly without

    # regex to avoid incorrect trimming, which would lead to the concatenation

    # of numbers. I would rather have missing values than incorrectly cleaned

    # numbers.

    try:

        # If it's not in square feet then I don't want to deal with all

        # possible conversions for now.

        if re.search(r"sq\.*\s*ft\.*", size) is None:

            return None

    

        size = size.replace(",", "")

        size = size.replace("'", "")

        size = size.replace("sq. ft.", "")

        size = size.replace("sf", "")

        size = size.strip()

        size = size.lower()

        

        add_mult_match = re.search(r"(\d+)\s*\+\s*(\d+)\s*(?:x|\*)\s*(\d+)", size)

        if add_mult_match is not None:

            return int(add_mult_match.groups()[0]) + (

                int(add_mult_match.groups()[1]) * 

                int(add_mult_match.groups()[2]))

        

        mult_match = re.search(r"(\d+)\s*(?:x|\*)\s*(\d+)", size)

        if mult_match is not None:

            return int(mult_match.groups()[0]) * int(mult_match.groups()[1])

        

        return int(size)

    # If any of the above doesn't work, just turn it into None/NaN

    # We want to guarantee this column is numeric

    except:

        return None

        

properties["Size Num"] = properties["Size Num"].apply(convert_size_num)
print("Properties with missing raw size data: {}".format(properties["Size"].isna().sum()))

print("Properties with missing size type data: {}".format(properties["Size Type"].isna().sum()))

print("Properties with missing size num data: {}".format(properties["Size Num"].isna().sum()))
properties.loc[properties["Size Num"].isna(), "Size Type"] = None
properties.loc[:, "Size Type"].value_counts(dropna=False)
properties["Bathrooms"].value_counts(dropna=False)
properties["Car Parks"].value_counts(dropna=False)
properties["Price per Area"] = properties["Price"] / properties["Size Num"]

properties["Price per Room"] = properties["Price"] / properties["Rooms Num"]
properties.to_csv("Properties_preprocessed.csv")
def plot_by_neighborhood(feature, formatting, factor=1):

    df = properties.groupby("Location")[feature].median().sort_values(ascending=False).reset_index()

    shift = 0.1 * (df[feature].max() - df[feature].min())

    df_sizes = properties.groupby("Location").size()[df["Location"]]



    fig = sns.catplot(

        data=df, x=feature, y="Location", kind="bar", 

        color="darkgrey", height=10, aspect=0.8)



    for index, row in df.iterrows():

        fig.ax.text(

            row[feature] + shift, row.name, formatting.format(row[feature] / factor), 

            color='black', ha="center", va="center")



    fig.ax.get_xaxis().set_visible(False);

    fig.despine(left=True, bottom=True)

    fig.ax.tick_params(left=False, bottom=False);

    fig.set_ylabels("");
plot_by_neighborhood(feature="Price", formatting="RM {:.2f}m", factor = 1e6)
plot_by_neighborhood(feature="Price per Area", formatting="RM {:.2f}k", factor = 1e3)
plot_by_neighborhood(feature="Price per Room", formatting="RM {:.2f}k", factor = 1e3)
plot_by_neighborhood(feature="Size Num", formatting="{:.2f}k sq. ft.", factor = 1e3)
plot_by_neighborhood(feature="Rooms Num", formatting="{:.2f}", factor = 1)
df = properties.groupby("Location").size().sort_values(ascending=False).reset_index()

shift = 0.05 * (df[0].max() - df[0].min())

df_sizes = properties.groupby("Location").size()[df["Location"]]



fig = sns.catplot(

    data=df, x=0, y="Location", kind="bar", 

    color="darkgrey", height=10, aspect=0.8)



for index, row in df.iterrows():

    fig.ax.text(

        row[0] + shift, row.name, row[0], 

        color='black', ha="center", va="center")



fig.ax.get_xaxis().set_visible(False);

fig.despine(left=True, bottom=True)

fig.ax.tick_params(left=False, bottom=False);

fig.set_ylabels("");
# Extract property type and turn it into a two-column data frame

df = properties.loc[~properties["Property Type Supergroup"].isna()].groupby(

    "Location")["Property Type Supergroup"].value_counts()

df.name = "Value"

df = df.reset_index().pivot(index="Location", columns="Property Type Supergroup")

df.columns = df.columns.droplevel(0)

df = df.fillna(0)



# normalize rows to see relative amount of properties in each neighborhood 

df_norm = df.apply(lambda x: x / x.sum(), axis=1)



fix, ax = plt.subplots(figsize=(12, 12))

hmap = sns.heatmap(

    df_norm, square=True, vmin=0, cmap="Reds", ax=ax, cbar=False)

hmap.set_ylabel(None);

hmap.set_xlabel(None);
df = properties[["Location", "Size Type", "Size Num"]].groupby(

    ["Location", "Size Type"]).median().reset_index()

fig = sns.catplot(

    data=df, x="Size Num", y="Location", kind="bar", 

    hue="Size Type", height=20, aspect=0.4);



fig.despine(left=True)

fig.ax.tick_params(left=False);

fig.set_ylabels("");
# Remove entries with "land area" in the "Size Type" column

Xy = properties.loc[properties["Size Type"] == "Built-up"]



# Keep only the relevant features

Xy = Xy.loc[:, [

    "Location", "Bathrooms", "Car Parks", "Furnishing", 

    "Rooms Num", "Property Type Supergroup", "Size Num", 

    "Price", "Price per Area", "Price per Room"]]



# Fill missing Car Parks feature values

Xy.loc[:, "Car Parks"] = Xy["Car Parks"].fillna(0)



# Remove entries with missing values

Xy = Xy.loc[Xy.isna().sum(axis=1) == 0]



# Specifically remove entries with "Unknown" furnishing status

Xy = Xy.loc[Xy["Furnishing"] != "Unknown"]



# Convert to dummy features

Xy = pd.get_dummies(Xy)
print("Shape of data frame: {}".format(Xy.shape))
print("Data frame DTYPES:")

for dtype in Xy.dtypes.unique():

    print(" - {}".format(dtype))
Xy["Size Num"].sort_values().head(10)
Xy["Size Num"].sort_values(ascending=False).head(20)
Xy = Xy.loc[Xy["Size Num"].between(250, 20000)]
selectors = []

for feature in ["Bathrooms", "Car Parks", "Rooms Num"]:

    selectors.append(Xy[feature].between(

        Xy[feature].quantile(0.001), 

        Xy[feature].quantile(0.999)))



Xy = Xy.loc[(~pd.DataFrame(selectors).T).sum(axis=1) == 0]
Xy, Xy_feature_selection = sklearn.model_selection.train_test_split(

    Xy, test_size=0.25, random_state=101)
Xy.shape
Xy_feature_selection.shape
fig, ax = plt.subplots(2, 2, figsize=(10, 10));

sns.countplot(data=Xy_feature_selection, x="Bathrooms", ax=ax[0, 0], color="darkgrey");

ax[0, 0].set_title("Bathrooms");

sns.countplot(data=Xy_feature_selection, x="Car Parks", ax=ax[0, 1], color="darkgrey");

ax[0, 1].set_title("Car Parks");

sns.countplot(data=Xy_feature_selection, x="Rooms Num", ax=ax[1, 0], color="darkgrey");

ax[1, 0].set_title("Rooms Num");

sns.distplot(a=Xy_feature_selection["Size Num"], bins=50, ax=ax[1, 1], color="darkgrey");

ax[1, 1].set_title("Size Num");
cols = ["Bathrooms", "Car Parks", "Rooms Num", "Size Num"]

Xy_feature_selection[cols] = sklearn.preprocessing.MinMaxScaler().fit_transform(

    Xy_feature_selection[cols])

Xy[cols] = sklearn.preprocessing.MinMaxScaler().fit_transform(Xy[cols])
hm_cmap = sns.diverging_palette(240, 0, s=99, l=50, as_cmap=True)

df = Xy_feature_selection[["Bathrooms", "Car Parks", "Rooms Num", "Size Num"]].corr()

sns.heatmap(data=df, vmin=-1, vmax=1, cmap=hm_cmap, annot=df, annot_kws={"size": 20});
Xy = Xy.drop(["Bathrooms", "Rooms Num"], axis=1)

Xy_feature_selection = Xy_feature_selection.drop(["Bathrooms", "Rooms Num"], axis=1)
df = Xy_feature_selection[["Price", "Price per Area", "Price per Room"]].corr()

sns.heatmap(

    df, vmin=-1, vmax=1, cmap=hm_cmap, 

    annot=np.round(df, 2), annot_kws={"size": 20})
Xy = Xy.drop("Price per Room", axis=1)

Xy_feature_selection = Xy_feature_selection.drop("Price per Room", axis=1)
Xy_train, Xy_test = sklearn.model_selection.train_test_split(Xy, test_size=0.2, random_state=101)

X_train = Xy_train.drop(["Price", "Price per Area"], axis=1)

y_train = Xy_train[["Price", "Price per Area"]]

X_test = Xy_test.drop(["Price", "Price per Area"], axis=1)

y_test = Xy_test[["Price", "Price per Area"]]
def train_and_test_model(

        model, X_train=X_train, y_train=y_train, 

        X_test=X_test, y_test=y_test, **kwargs):

    model.fit(X_train, y_train, **kwargs)

    y_pred = model.predict(X_test)

    r2 = sklearn.metrics.r2_score(y_true=y_test, y_pred=y_pred)

    return model, r2
model, r2 = train_and_test_model(

    model = sklearn.linear_model.LinearRegression(), 

    X_train=X_train, y_train=y_train["Price"], 

    X_test=X_test, y_test=y_test["Price"])

print("R^2 for prediction of 'Price': {:.2f}".format(r2))



model, r2 = train_and_test_model(

    model = sklearn.linear_model.LinearRegression(), 

    X_train=X_train, y_train=y_train["Price per Area"], 

    X_test=X_test, y_test=y_test["Price per Area"])

print("R^2 for prediction of 'Price per Area': {:.2f}".format(r2))
def make_fcn_model():

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(units=32, activation="relu", input_shape=(X_train.shape[1],)))

    model.add(keras.layers.Dense(units=32, activation="relu"))

    model.add(keras.layers.Dense(units=32, activation="relu"))

    model.add(keras.layers.Dense(units=1, activation="relu"))

    model.compile(loss="mse", optimizer="Adam")

    return model
model, r2 = train_and_test_model(

    model = make_fcn_model(), 

    X_train=X_train, y_train=y_train["Price"], 

    X_test=X_test, y_test=y_test["Price"], 

    batch_size=8, epochs=10, verbose=0)

print("R^2 for prediction of 'Price': {:.2f}".format(r2))



model, r2 = train_and_test_model(

    model = make_fcn_model(), 

    X_train=X_train, y_train=y_train["Price per Area"], 

    X_test=X_test, y_test=y_test["Price per Area"], 

    batch_size=8, epochs=10, verbose=0)

print("R^2 for prediction of 'Price per Area': {:.2f}".format(r2))
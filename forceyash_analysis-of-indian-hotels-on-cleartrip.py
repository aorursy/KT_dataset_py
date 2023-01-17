# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/cleartrip-comtravel-samplecsv"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from mpl_toolkits.basemap import Basemap

from matplotlib import pyplot as plt

from scipy import stats

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer

from statistics import mode

from collections import Counter

import warnings

import re

from sklearn.cluster import KMeans  



warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/cleartrip-comtravel-samplecsv/cleartrip_com-travel_sample.csv")
df["tad_review_count"] = df["tad_review_count"].fillna(0)

df["review_value"] = df["tad_review_count"] * df["tad_review_rating"]

df["property_type"] = df["property_type"].fillna("uncategorized")

df["hotel_description"] = df["hotel_description"].fillna("no description")

df["similar_hotel"] = df["similar_hotel"].fillna(0)



i=0

property_type = df.property_type.unique()



for t,x in zip(df["property_type"], df["hotel_description"]):

    if ((t == "uncategorized") & (x != "no description")): 

        for l in nltk.word_tokenize(x):

            if l in property_type:

                try:

                    df.at[i , "property_type"] = l

                    break

                except:

                    print(i)

                            

    i=i+1
i=0

for x,y in zip(df["property_type"], df["similar_hotel"]):

    prop_type = []

    if ((x == "uncategorized") & (y != 0)):

        same_props = y.split("|")

        for z in same_props:

            try:

                cat = df.where(df["property_id"] == int(strip(z)))["property_type"]

                if (cat != "uncategorized"):

                    prop_type.append(cat)

            except:

                pass

        try:

            df.at[i , "property_type"] = mode(prop_type)

        except:

            if len(prop_type) != 0:

                df.at[i , "property_type"] = prop_type[0]

        

    i=i+1
df[["city", "hotel_star_rating"]].where(df["hotel_star_rating"] == "5 Star hotel")["city"].value_counts().nlargest(5)
df[["state", "hotel_star_rating"]].where(df["hotel_star_rating"] == "5 Star hotel")["state"].value_counts().nlargest(5)
print(df["property_type"].unique())

df["property_type"].value_counts()
for i in df["property_type"].dropna().unique():

    avg = df[["property_type", "room_count"]].where(df["property_type"] == i)["room_count"].sum()/df.where(df["property_type"] == i)["room_count"].count()

    print(i, ": ", np.round(avg))
big = df[df.room_count == df.room_count.max()]

fig = plt.figure(figsize=(20, 5))

m = Basemap(projection='lcc',width=8E6, height=10E6, lat_0 = 20, lon_0 = 79, 

            llcrnrlat = 5, llcrnrlon = 65, urcrnrlat = 40, urcrnrlon=100, resolution = "l")

m.drawcoastlines()

m.drawcountries()

m.shadedrelief()

# Map (long, lat) to (x, y) for plotting

x, y = m(big['longitude'].values, big['latitude'].values)

plt.plot(x, y, 'r*', markersize=15)

plt.text(x, y, big.iloc[0]["property_name"], fontsize=12)

#need to remove outliers first

avg_rooms_df = df[["property_type", "room_count"]]

avg_rooms_df["z_score"] = np.abs(stats.zscore(avg_rooms_df["room_count"])) < 3

avg_rooms_df = avg_rooms_df[avg_rooms_df.z_score]



plt.subplots(figsize = (12,6))

sns.boxplot(x = "property_type", y = "room_count", data = avg_rooms_df)

plt.show()

sns.swarmplot(x = "property_type", y = "room_count", data = avg_rooms_df, palette = "Set1")

fig = plt.gcf()

fig.set_size_inches(12,8)

plt.ylim(-5,avg_rooms_df["room_count"].max() + 15)

plt.show()
outliers = df[["property_type", "room_count", "longitude", "latitude", "property_name"]]

outliers["z_score"] = np.abs(stats.zscore(outliers["room_count"])) > 3

outliers = outliers[outliers.z_score]

fig = plt.figure(figsize=(80, 20))

m = Basemap(projection='lcc',width=8E6, height=10E6, lat_0 = 20, lon_0 = 79, 

            llcrnrlat = 5, llcrnrlon = 65, urcrnrlat = 40, urcrnrlon=100, resolution = "l")

m.drawcoastlines()

m.drawcountries()

m.shadedrelief()

# Map (long, lat) to (x, y) for plotting

for a,b,c in zip(outliers['longitude'],  outliers['latitude'], outliers['property_name']):

    x, y = m(a,b)

    plt.plot(x, y, 'r*', markersize=15)

    plt.text(x, y, c, fontsize=12)
hotel_desc = df["hotel_description"].str.cat(sep = " ")

hotel_desc = hotel_desc.lower()

list_of_word = []

tag = []

def process_content():

    try:

        words = nltk.word_tokenize(hotel_desc)

        tagged = nltk.pos_tag(words)

        for i in tagged:

            list_of_word.append(i[0])

            tag.append(i[1])

            #list_of_word.append(tagged[:,0])

    except Exception as e:

        print(str(e))



process_content()



all_words = pd.DataFrame(data = {"words": list_of_word, "tag": tag})

all_adj = all_words[(all_words.tag == "JJ") | (all_words.tag == "JJR") | (all_words.tag == "JJS")]

all_adj["words"].value_counts().nlargest(20)
#all_verbs = all_words[(all_words.tag == "VB") | (all_words.tag == "VBD") | (all_words.tag == "VBG") 

#                      | (all_words.tag == "VBN")| (all_words.tag == "VBP")| (all_words.tag == "VBZ")]

all_verbs = all_words[(all_words.tag == "VBG") | (all_words.tag == "VBP")| (all_words.tag == "VBZ")]



all_verbs["words"].value_counts().nlargest(10)
prop_name = df["property_name"].str.cat(sep = " ")

prop_name = prop_name.lower()

#custom_sent_tokenizer = PunktSentenceTokenizer()

#tokenized = custom_sent_tokenizer.tokenize(prop_name)

list_of_names = []

tags = []



def process_content():

    try:

        words = nltk.word_tokenize(prop_name)

        tagged = nltk.pos_tag(words)

        for i in tagged:

            list_of_names.append(i[0])

            tags.append(i[1])

    except Exception as e:

        print(str(e))



process_content()



all_hotel_names = pd.DataFrame(data = {"words": list_of_names, "tag": tags})

all_proper_nouns = all_hotel_names[(all_hotel_names.tag == "NNP") | (all_hotel_names.tag == "NN") | (all_hotel_names.tag == "NNS")]

all_proper_nouns["words"].value_counts()[:30]
df[["city", "hotel_star_rating", "property_name", "property_type", "tad_review_count"]].nlargest(10, "tad_review_count")
df[["city", "hotel_star_rating", "property_name", "property_type", "review_value"]].nlargest(10, "review_value")
prop_type = df[["property_type", "room_facilities"]].where(df["property_type"] == "uncategorized")

fecilities = prop_type["room_facilities"].str.cat(sep = " | ")

fecilities = fecilities.lower()

fecilities = fecilities.split(sep = "|")

fecilities = [x.strip() for x in fecilities]

mst_com_fes = [i[0] for i in Counter(fecilities).most_common(10)]

print(mst_com_fes)
for fes in df["property_type"].unique():

    if (fes != "uncategorized"):

        prop_type = df[["property_type", "room_facilities"]].where(df["property_type"] == fes)

        fecilities = prop_type["room_facilities"].str.cat(sep = " | ")

        fecilities = fecilities.lower()

        fecilities = fecilities.split(sep = "|")

        fecilities = [x.strip() for x in fecilities]

        print("Top 10 most common room fecilities in ", fes, ": ")

        mst_com_fes = [i[0] for i in Counter(fecilities).most_common(5)]

        print(mst_com_fes, "\n")

        
data = df[["longitude", "latitude"]].dropna()

fig = plt.figure(figsize=(60, 15))

m = Basemap(projection='mill')#,width=8E6, height=10E6, lat_0 = 20, lon_0 = 79, 

            #llcrnrlat = 5, llcrnrlon = 65, urcrnrlat = 40, urcrnrlon=100, resolution = "l")

m.drawcoastlines()

m.drawcountries()

m.shadedrelief()

# Map (long, lat) to (x, y) for plotting

x, y = m(data['longitude'].values, data['latitude'].values)

plt.plot(x, y, 'o', markersize=5)

#plt.text(x, y, 'Delhi', fontsize=12)
data = df[["longitude", "latitude"]].dropna()

data = data.where(~(data.latitude >= 37.6) & ~(data.latitude <= 8.4)).dropna()

data = data.where((data["longitude"] <= 97.25) & (data["longitude"] >= 68.7)).dropna()
coordinates= np.array([list((a,float(b))) for a,b in zip(data['longitude'], data['latitude'])])

x = [t for t in range(1,30)]

y = []

for k in range(1,30):

    kmeans = KMeans(n_clusters = k, random_state=1).fit(coordinates)

    labels = kmeans.labels_

    interia = kmeans.inertia_

    #print("k:",k, " cost:", interia)

    y.append(interia)

    

sns.lineplot(x=x, y=y)
fig = plt.figure(figsize=(60, 15))

m = Basemap(projection='mill',width=8E6, height=10E6, lat_0 = 20, lon_0 = 79, 

            llcrnrlat = 5, llcrnrlon = 65, urcrnrlat = 40, urcrnrlon=100, resolution = "l")



m.drawcoastlines()

m.drawcountries()

m.shadedrelief()



coordinates= np.array([list((a,float(b))) for a,b in zip(data['longitude'], data['latitude'])])

kmeans = KMeans(n_clusters = 10)

kmeans.fit(coordinates)

x, y = m(coordinates[:,0],coordinates[:,1])

plt.scatter(x,y,c=kmeans.labels_.astype(float))

centers = kmeans.cluster_centers_

x, y = m(centers[:, 0], centers[:, 1])

plt.scatter(x, y, c='black', s=200, alpha=0.5);

plt.show()
cluster_map = pd.DataFrame()

cluster_map['data_index'] = data.index.values

cluster_map['cluster'] = kmeans.labels_

new_df = df

new_df = new_df.reset_index().rename(columns  = {"index": "data_index"})#.drop(columns = "level_0")

new_df= pd.merge(new_df, cluster_map, how = 'inner', on = "data_index")

sns.countplot(x="cluster", hue = "property_type", data = new_df)

print(new_df["cluster"].value_counts())

min_clust = new_df["cluster"].value_counts().reset_index().iloc[-3:]["index"]

fig = plt.figure(figsize=(30, 8))

m = Basemap(projection='mill',width=8E6, height=10E6, lat_0 = 20, lon_0 = 79, 

            llcrnrlat = 5, llcrnrlon = 65, urcrnrlat = 38, urcrnrlon=100, resolution = "l")



m.drawcoastlines()

m.drawcountries()

m.shadedrelief()



min_clus = new_df[new_df.cluster.isin(min_clust)]

# Map (long, lat) to (x, y) for plotting

x, y = m(min_clus['longitude'].values, min_clus['latitude'].values)

#plt.plot(x, y, 'o', markersize=5)

plt.scatter(x,y,c=min_clus.cluster)

avg_rooms_clus = min_clus[["property_type", "room_count", "hotel_star_rating", "cluster"]]

avg_rooms_clus["z_score"] = np.abs(stats.zscore(avg_rooms_clus["room_count"])) < 3

avg_rooms_clus = avg_rooms_clus[avg_rooms_clus.z_score]



plt.subplots(figsize = (12,6))

sns.boxplot(x = "property_type", y = "room_count", data = avg_rooms_clus)

plt.show()

sns.swarmplot(x = "property_type", y = "room_count", hue = "hotel_star_rating", data = avg_rooms_clus, palette = "Set1")

fig = plt.gcf()

fig.set_size_inches(12,8)

plt.ylim(-5,avg_rooms_clus["room_count"].max() + 15)

plt.show()
avg_rooms_clus.groupby(["hotel_star_rating", "cluster"]).property_type.count()
fig = plt.figure(figsize=(30, 8))

m = Basemap(projection='mill',width=8E6, height=10E6, lat_0 = 20, lon_0 = 79, 

            llcrnrlat = 5, llcrnrlon = 65, urcrnrlat = 38, urcrnrlon=100, resolution = "l")



m.drawcoastlines()

m.drawcountries()

m.shadedrelief()



min_clus = new_df[new_df.cluster == 4]

# Map (long, lat) to (x, y) for plotting

x, y = m(min_clus['longitude'].values, min_clus['latitude'].values)

#plt.plot(x, y, 'o', markersize=5)

plt.scatter(x,y,c=min_clus.cluster)

min_clus["city"].value_counts()
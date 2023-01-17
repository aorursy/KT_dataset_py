# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Libraries for plotting

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")





#to count occurences of attributes in data

from collections import Counter



# Input data files are available in the "../input/" directory.



import os



#to look at test and train images import

import cv2
train_df = pd.read_csv("/kaggle/input/imet-2020-fgvc7/train.csv")

train_df.head()
labels_df = pd.read_csv("../input/imet-2020-fgvc7/labels.csv")

labels_df.head()
train_path = "../input/imet-2020-fgvc7/train/" #all images are .png
train_df['Number of tags']= train_df['attribute_ids'].apply(lambda x:len(x.split(' ')))



tags = sns.countplot(x='Number of tags',data=train_df ,palette='rocket')

plt.ylabel('Number of images')

plt.title("How many tags were used to describe image?")

sns.despine()



fig_3 = tags.get_figure()

fig_3.savefig("tag_num.png")
#this code line to separate the original attributes which where attribute_type::attribute_name

#throws an error if run more than once so comment it out once the column has been split



attributes = labels_df["attribute_name"]

labels_df[["attribute_type", "attribute_name"]] = attributes.str.split("::", expand=True)
#this is meant to connect the attribute names with the numerical ids of the attributes in the training data for visualization purposes



#this flattens the train.csv into a one dimensional array of all ids assigned to all images

id_perImg = train_df["attribute_ids"].to_numpy()

ids = []

for id_list in id_perImg:

    id_list = id_list.split(" ")

    for id in id_list:

        id = int(id)

        ids.append(id)
#counts how many times each id appears in the flattened list

occurences = Counter(ids)

occ = sorted(occurences.items())



#created a dataframe of the ids and count so you can merge it with the df containing meaningful names

occ_df = pd.DataFrame(occ, columns=['attribute_id','count'])

counts_df = pd.merge(occ_df, labels_df, on='attribute_id', how= "right")

counts_df = counts_df.sort_values(by='attribute_id', ascending=True).set_index("attribute_id")



#occ_df has 3471 rows × 2 columns meaning that 3 attribute ids haven't been used to identify artworks

counts_df.head()
top30_att = counts_df.sort_values(by="count",ascending=False)

top30_att.iloc[0:30,0]



sns.set(font_scale = 3)

sns.set_style("whitegrid")



plt.subplots(figsize=(60,25))

ax = sns.barplot(y=top30_att.iloc[0:30,1],x=top30_att.iloc[0:30,0], palette='rocket')

plt.ylabel('Attributes')

plt.xlabel('Count')

plt.title("The top 25 out of 3474 attributes used to label the images")

sns.despine()

fig_all = ax.get_figure()

fig_all.savefig("allLabels.png",transparent=True)

#Code segment to show how many times the attribute types where used to describe the artwork in the train data



medium_occ = counts_df.loc[counts_df['attribute_type'] == "medium", 'count'].sum()

tag_occ = counts_df.loc[counts_df['attribute_type'] == "tags", 'count'].sum()

culture_occ = counts_df.loc[counts_df['attribute_type'] == "culture", 'count'].sum()

country_occ = counts_df.loc[counts_df['attribute_type'] == "country", 'count'].sum()

dimension_occ = counts_df.loc[counts_df['attribute_type'] == "dimension", 'count'].sum()



occurences_atypes = [["medium","tags", "culture","country","dimension"],[medium_occ, tag_occ, culture_occ, country_occ, dimension_occ]]
sns.set(font_scale = 1)

sns.set_style("whitegrid")



#2 plots to understand the attributes and their distribution better

dist_att = sns.barplot(labels_df["attribute_type"].value_counts().index, labels_df["attribute_type"].value_counts(), palette='rocket')

dist_att.set(xlabel="Grouped by attribute types", ylabel='Number of existing attributes')

plt.title("How many distinct attributes exist per attribute group?")

plt.show()



fig_2 = dist_att.get_figure()

fig_2.savefig("attribs.png",transparent=True)



occ = sns.barplot(occurences_atypes[0] ,occurences_atypes[1], palette='rocket')

occ.set(xlabel="Grouped by attribute types", ylabel='Occurence of attribute type in train set')

plt.title("How often did experts assign different attribute types to the train set?")

plt.show()

fig_3 = occ.get_figure()

fig_3.savefig("occ_attribs.png",transparent=True)

#This segment splits the ids into their subgroups for later visualization

#Also in the previous year it was mentioned that the culture and country labels tend to be noisy and it might be good to train them seperately



country_df = counts_df[counts_df.attribute_type.values == 'country']

culture_df = counts_df[counts_df.attribute_type.values == 'culture']

dimension_df = counts_df[counts_df.attribute_type.values == 'dimension']

medium_df = counts_df[counts_df.attribute_type.values == 'medium']

tags_df = counts_df[counts_df.attribute_type.values == 'tags']
#explore if every artwork has a culture or country associated with it 

# countries are all ids between 0 and 99

# cultures are all ids between 100 and 780



all_ids = train_df.iloc[:,1].to_numpy()

id_lists = [elem.split(" ") for elem in all_ids]

country_culture = [[el for el in id_list if int(el) < 781] for id_list in id_lists]

country_culture[0:10]



#judging by this not every artwork was assigned a culture or country label and some were assigned 2 or more culture/country labels therefore they are not mutually exclusive
# This will be the segment for some bar charts so we can see how the labels are distributed

# Ideally we could figure out how to display the count next to the bar so that it makes the scale difference from one image to the next more clear

sns.set_style("whitegrid")



country_vis = country_df.sort_values(by="count", ascending=False) 



plt.figure(figsize=(15, 5))

top20 = country_vis.head(n=10)

country = sns.barplot(top20["count"],top20["attribute_name"], palette='rocket')

country.set(xlabel="Count per label", ylabel='Country label')

plt.title("Top 10 assigned country labels out of 100 total")

plt.show()



fig_4 = country.get_figure()

fig_4.savefig("occ_countries.png",transparent=True)



# I separated it into 2 segments so you could see what counts the bottom 50 have

# Since they were so much smaller quantities than in the top 30 there seemed to be no occurences at all in a plot showing all at once

#rest = country_vis.tail(n=80)

#plt.figure(figsize=(16, 20))

#rest = sns.barplot(rest["count"],rest["attribute_name"] )

#rest.set(xlabel="Count of label", ylabel='Country label')

#rest.xaxis.set_label_position('top') 

#rest.xaxis.tick_top()

#plt.title("The rest of the counts are far smaller ")

#plt.show()
culture_vis = culture_df.sort_values(by="count", ascending=False)

plt.figure(figsize=(16, 7))

top20 = culture_vis.head(n=15)

culture = sns.barplot(top20["count"],top20["attribute_name"], palette='rocket')

culture.set(xlabel="Count per label", ylabel='Culture label')

plt.title("Top 15 assigned culture labels out of 681 cultures")

plt.show()



fig_5 = culture.get_figure()

fig_5.savefig("occ_cultures.png",transparent=True)
dimension_vis = dimension_df.sort_values(by="count", ascending=False)



dim = sns.barplot(dimension_vis["attribute_name"],dimension_vis["count"], palette='rocket')

dim.set(xlabel='Dimension labels', ylabel="Count per Dimension")

plt.title("All Assigned dimension labels")

plt.show()



fig_6 = dim.get_figure()

fig_6.savefig("occ_dim.png",transparent=True)
medium_vis = medium_df.sort_values(by="count", ascending=False)

plt.figure(figsize=(16, 7))

top20_m = medium_vis.head(n=15)

med = sns.barplot(top20_m["count"],top20_m["attribute_name"], palette='rocket')

med.set(xlabel="Count per label", ylabel='Medium label')

plt.title("Top 15 assigned medium labels from 1920 labels")

plt.show()



fig_7 = med.get_figure()

fig_7.savefig("occ_med.png",transparent=True)
tags_vis = tags_df.sort_values(by="count", ascending=False)

plt.figure(figsize=(16, 7))

top20_t = tags_vis.head(n=15)

tags = sns.barplot(top20_t["count"],top20_t["attribute_name"], palette='rocket')

tags.set(xlabel="Count of label", ylabel='Tag label')

plt.title("Top 15 assigned tags from 768 total tags")

plt.show()



fig_8 = tags.get_figure()

fig_8.savefig("occ_tags.png",transparent=True)
#visualize test images

count = 1

plt.figure(figsize=[16,16])



for img_name in os.listdir("../input/imet-2020-fgvc7/test/")[:16]:

    img = cv2.imread("../input/imet-2020-fgvc7/test/{}".format(img_name))[...,[2,1,0]]

    plt.subplot(4,4,count)

    plt.imshow(img)

    plt.title("test image {}".format(count))

    count += 1

plt.show();
#visualize train images

sns.set_style('white')

plt.figure(figsize=[22,20])

count=1

for img_name in os.listdir('../input/imet-2020-fgvc7/train/')[:36]:

    img = cv2.imread('../input/imet-2020-fgvc7/train/%s'%img_name)

    plt.subplot(6,6,count)

    plt.imshow(img)

    plt.title('Item %s'%count)

    count+=1
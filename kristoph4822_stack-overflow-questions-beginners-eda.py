import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import re

from collections import Counter
df = pd.read_csv("../input/60k-stack-overflow-questions-with-quality-rate/data.csv",index_col = "Id")

df.head()
df.shape
df.groupby(["Y"]).Y.count().plot(kind='bar')
# returns dict with each tag appearance in Tags column counted

def tags_counter(tags):

        

    tag_dict = {}

    for index, value in tags.items(): # iterating over Tags column 

        tag_list = re.findall(r'\w+', value) # extracting tags into list

        for tag in tag_list:

            if tag in tag_dict.keys():

                tag_dict[tag] = tag_dict[tag] + 1

            else:

                tag_dict[tag] = 1

    

    return tag_dict
# creates bar plot with n most popular tags, title parameter for the plot

def print_top_tags(tag_dict, n, title):

    tag_dict = dict(Counter(tag_dict).most_common(n))

    fig= plt.figure(figsize=(12,6))

    plt.bar(range(n), list(tag_dict.values()), align='center')

    plt.xticks(range(n), list(tag_dict.keys()))

    plt.title("Top " + str(n) + " most popular tags " + title) 

    plt.show()
n = 10 # how many top tags we want to see

    

print_top_tags(tags_counter(df.Tags), n, "overall") 
q_list = list(df.Y.unique()) # list of quality categories



for q in q_list:

    title = "for " + q + " posts"

    print_top_tags(tags_counter(df[df.Y == q].Tags), n, title) 
# import import import I love import

import numpy as np

import pandas as pd

import re

import seaborn as sns

import matplotlib.pyplot as plt
# Read in CSV to dataframe

df = pd.read_csv('../input/hacker_news_sample.csv')
# Examine the fields

df.columns
# Use regex to get the posts with cuss words

num = len(df["text"])

cuss_post = 0

authors = {}

for i in range(0,num):

    has_cuss = re.search(r"shit|fuck|damn", str(df["text"][i]))

    if has_cuss:

        cuss_post += 1

        author = str(df["by"][i])

        if author in authors.keys():

            authors[author] += 1

        else:

            authors[author] = 1
print("Out of", num, "posts,", cuss_post, "posts have cuss words", '(' + str(round(cuss_post/num*100, 2)) + '%)')
dff = pd.DataFrame.from_dict(authors, orient="index").reset_index()

dff.columns=['authors', 'num_posts']

dff.head(20)
dff.loc[dff["num_posts"].max()]

data = dff.sort_values("num_posts", ascending=False).head(10)

data
viz = sns.barplot(y=data.authors, x=data.num_posts, orient="h", palette="Set1")

viz.set(xlabel="Number of posts with cuss words", ylabel="Author")

plt.show()
cuss_list = ["shit", "shitter", "fuck", "fucked", "fucking", "fucker", "damn", "damnit", "ass", "asshole"]



df_cuss = pd.DataFrame({"by": df.by, "text": df.text})

df_cuss.head()
# This method isn't perfect. I should change this to use regex instead 

def count_cusses(text):

    cuss_list = ["shit", "shitter", "fuck", "fucked", "fucking", "fucker", "damn", "damnit", "ass", "asshole"]

    text = str(text)

    count = 0

    for word in text.split(" "):

        if word in cuss_list:

            count += 1

    return count



# Count the cusses in each post

df_cuss["total_cusses"] = df_cuss.text.apply(count_cusses)



# Only want posts with cuss words

df_cuss = df_cuss[df_cuss.total_cusses > 0]
# Authors with the most cusses total

df_cuss.groupby("by").sum().sort_values("total_cusses", ascending=False)[:20]
# Top 20 posts with the most cusses

df_cuss.sort_values("total_cusses", ascending=False)[:20]
# Top 20 posts with the most cusses per word

def cusses_per_word(text):

    text = str(text)

    words_in_text = len(text.split(" "))

    num_cusses = count_cusses(text)

    return (num_cusses/words_in_text)



df_cuss["cusses_per_word"] = df_cuss.text.apply(cusses_per_word)



df_cuss.sort_values("cusses_per_word", ascending=False)[:20]
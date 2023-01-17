# We will use this libraries

import numpy as np

import pandas as pd

import nltk



from PIL import Image

import urllib.request

from collections import Counter

from wordcloud import WordCloud, ImageColorGenerator



import matplotlib.pyplot as plt

import plotly.graph_objects as go
# Reading data from csv

dataFrame = pd.read_csv("/kaggle/input/covid19-250000-tweets/covid19_en.csv")
# This functions gets a tweet as string and finds hashtags or usernames



def extractHashTags(tweet):

    # We use nltk TweetTokenizer to tokenize twitter special keywords like #abc or @abc

    tokenized_tweet = nltk.tokenize.TweetTokenizer().tokenize(tweet)

    # We check every token/word first char

    hashTags = [word for word in tokenized_tweet if word[0] == "#" and len(word) > 1]

    return hashTags



def extractUserNames(tweet):

    tokenized_tweet = nltk.tokenize.TweetTokenizer().tokenize(tweet)

    userNames = [word for word in tokenized_tweet if word[0] == "@" and len(word) > 1]

    return userNames
# We apply functions to our dataframe. Results will write as new columns

dataFrame["userNames"] = dataFrame["tweet"].apply(extractUserNames)

dataFrame["hashTags"] = dataFrame["tweet"].apply(extractHashTags)
# Filter if there is no hashtag or username in a tweet

userNames = dataFrame[dataFrame["userNames"].str.len() != 0]['userNames']

hashTags = dataFrame[dataFrame["hashTags"].str.len() != 0]['hashTags']
# Merge lists as a one new list

uN = [i for u in userNames.values for i in u]

hT = [i for h in hashTags.values for i in h]
# To find most frequent elements in a list we need this function

def mostFrequentelemnts(List, n): 

    occurence_count = Counter(List) 

    return occurence_count.most_common(n)
# We use plotly bar graph to visualize most used 10 usernames



fig = go.Figure()

fig.add_trace(go.Bar(

    x=[i[0] for i in mostFrequentelemnts(uN, 10)],

    y=[i[1] for i in mostFrequentelemnts(uN, 10)],

    marker_color='aqua',

    opacity=0.5

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(title = "Most Used 10 Usernames", barmode='group', xaxis_tickangle=-45)

fig.show()
# Create our images and show

mask = np.array(Image.open(urllib.request.urlopen("https://dslv9ilpbe7p1.cloudfront.net/_FsA5Yg9iTCQnkQFlfyrxw_store_header_image")))



wc = WordCloud(background_color="black", width = 1920, height = 1080)



wc.generate(" ".join(uN))



image_colors = ImageColorGenerator(mask)

wc.recolor(color_func=image_colors)



fig = plt.figure(figsize = (40, 30))

plt.imshow(wc, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
wc = WordCloud(background_color="black", width = 1920, height = 1080)



wc.generate(" ".join(hT))



image_colors = ImageColorGenerator(mask)

wc.recolor(color_func=image_colors)



fig = plt.figure(figsize = (40, 30))

plt.imshow(wc, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
filtered_hT = [hashtag for hashtag in hT if "covid" not in hashtag.lower() and "corona" not in hashtag.lower()]
fig = go.Figure()

fig.add_trace(go.Bar(

    x=[i[0] for i in mostFrequentelemnts(filtered_hT, 10)],

    y=[i[1] for i in mostFrequentelemnts(filtered_hT, 10)],

    marker_color='magenta',

    opacity=0.5

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(title = "Most Used 10 Hashtags", barmode='group', xaxis_tickangle=-45)

fig.show()
wc = WordCloud(background_color="black", width = 1920, height = 1080)



wc.generate(" ".join(filtered_hT))



image_colors = ImageColorGenerator(mask)

wc.recolor(color_func=image_colors)



fig = plt.figure(figsize = (40, 30))

plt.imshow(wc, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
dataFrame = pd.read_csv("/kaggle/input/covid19-250000-tweets/covid19_tr.csv", encoding='utf-8')



# We apply functions to our dataframe. Results will write as new columns

dataFrame["userNames"] = dataFrame["tweet"].apply(extractUserNames)

dataFrame["hashTags"] = dataFrame["tweet"].apply(extractHashTags)



# Filter if there is no hashtag or username in a tweet

userNames = dataFrame[dataFrame["userNames"].str.len() != 0]['userNames']

hashTags = dataFrame[dataFrame["hashTags"].str.len() != 0]['hashTags']



# Merge lists as a one new list

uN = [i for u in userNames.values for i in u]

hT = [i for h in hashTags.values for i in h]
fig = go.Figure()

fig.add_trace(go.Bar(

    x=[i[0] for i in mostFrequentelemnts(uN, 10)],

    y=[i[1] for i in mostFrequentelemnts(uN, 10)],

    marker_color='aqua',

    opacity=0.5

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(title = "Most Used 10 Usernames", barmode='group', xaxis_tickangle=-45)

fig.show()
wc = WordCloud(background_color="black", width = 1920, height = 1080)



wc.generate(" ".join(uN))



image_colors = ImageColorGenerator(mask)

wc.recolor(color_func=image_colors)



fig = plt.figure(figsize = (40, 30))

plt.imshow(wc, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
filtered_hT = [hashtag for hashtag in hT if "covid" not in hashtag.lower() and "corona" not in hashtag.lower()]
fig = go.Figure()

fig.add_trace(go.Bar(

    x=[i[0] for i in mostFrequentelemnts(filtered_hT, 10)],

    y=[i[1] for i in mostFrequentelemnts(filtered_hT, 10)],

    marker_color='magenta',

    opacity=0.5

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(title = "Most Used 10 Hashtags", barmode='group', xaxis_tickangle=-45)

fig.show()
wc = WordCloud(background_color="black", width = 1920, height = 1080)



wc.generate(" ".join(filtered_hT))



image_colors = ImageColorGenerator(mask)

wc.recolor(color_func=image_colors)



fig = plt.figure(figsize = (40, 30))

plt.imshow(wc, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
dataFrame = pd.read_csv("/kaggle/input/covid19-250000-tweets/covid19_de.csv", encoding='utf-8')



# We apply functions to our dataframe. Results will write as new columns

dataFrame["userNames"] = dataFrame["tweet"].apply(extractUserNames)

dataFrame["hashTags"] = dataFrame["tweet"].apply(extractHashTags)



# Filter if there is no hashtag or username in a tweet

userNames = dataFrame[dataFrame["userNames"].str.len() != 0]['userNames']

hashTags = dataFrame[dataFrame["hashTags"].str.len() != 0]['hashTags']



# Merge lists as a one new list

uN = [i for u in userNames.values for i in u]

hT = [i for h in hashTags.values for i in h]
fig = go.Figure()

fig.add_trace(go.Bar(

    x=[i[0] for i in mostFrequentelemnts(uN, 10)],

    y=[i[1] for i in mostFrequentelemnts(uN, 10)],

    marker_color='aqua',

    opacity=0.5

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(title = "Most Used 10 Usernames", barmode='group', xaxis_tickangle=-45)

fig.show()
wc = WordCloud(background_color="black", width = 1920, height = 1080)



wc.generate(" ".join(uN))



image_colors = ImageColorGenerator(mask)

wc.recolor(color_func=image_colors)



fig = plt.figure(figsize = (40, 30))

plt.imshow(wc, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
filtered_hT = [hashtag for hashtag in hT if "covid" not in hashtag.lower() and "corona" not in hashtag.lower()]
fig = go.Figure()

fig.add_trace(go.Bar(

    x=[i[0] for i in mostFrequentelemnts(filtered_hT, 10)],

    y=[i[1] for i in mostFrequentelemnts(filtered_hT, 10)],

    marker_color='magenta',

    opacity=0.5

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(title = "Most Used 10 Hashtags", barmode='group', xaxis_tickangle=-45)

fig.show()
wc = WordCloud(background_color="black", width = 1920, height = 1080)



wc.generate(" ".join(filtered_hT))



image_colors = ImageColorGenerator(mask)

wc.recolor(color_func=image_colors)



fig = plt.figure(figsize = (40, 30))

plt.imshow(wc, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
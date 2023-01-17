import os

import numpy as np

import pandas as pd

import re

from datetime import datetime

import seaborn as sns

import matplotlib.pyplot as plt

import plotly

import plotly.plotly as py

import plotly.graph_objs as go

import colorlover as cl

plotly.offline.init_notebook_mode() 
twitter_files = os.listdir("../input/twitter")

twitter_users_files = os.listdir("../input/twitter_users")

pic_files = os.listdir("../input/pics")

metadata = pd.read_csv("../input/candidates_info.csv")
def clean(text):



    text = re.sub(r'#\S*', ' ', text)  

    text = re.sub(r'http\S*', ' ', text)

    for ch in ['\\','`','*','_','{','}','[',']','(',')','>','+','-','.','!','\'',"\”",'\"', '\“', "\’", "?", ":",

               "-",",", "//t", "&amp;", "/", "'", "'", "…","-", "’", "\—", "—", "–", "“", "”"]:

        if ch in text:

            text = text.replace(ch," ")

    

    return(text)  



def pair_words(text):

    text = text.replace("climate change", "climate_change")

    text = text.replace("health care", "health_care")

    text = text.replace("we need", "we_need")

    text = text.replace("we must", "we_must")

    text = text.replace("we can t", "we_can_t")

    text = text.replace("we can", "we_can")

    text = text.replace("we have", "we_have")

    text = text.replace("we are", "we_are")

    text = text.replace("we re", "we_are")

    text = text.replace("thank you", "thank_you")

    text = text.replace("united states", "united_states")

    text = text.replace("american people", "american_people")

    text = text.replace("town hall", "town_hall")

    text = text.replace("gun violence", "gun_violence")

    text = text.replace("join us", "join_us")

    text = text.replace("looking forward", "looking_forward")

    text = text.replace("white house", "white_house")

    text = text.replace("right now", "right_now")

    text = text.replace("supreme court", "supreme_court")

    text = text.replace("new york", "new_york")

    text = text.replace("middle class", "middle_class")

    text = text.replace("south bend", "south_bend")

    text = text.replace("don t", "don_t")

    text = text.replace("for all", "for_all")

    text = text.replace("we will", "we_will")

    text = text.replace("join me", "join_me")

    text = text.replace("national security", "national_security")

    text = text.replace("bill weld", "bill_weld")

    text = text.replace("de blasio", "de_blasio")

    

    

    return(text)    





def clean_tweet(tweet):

    return ' '.join(pair_words(clean(tweet.lower())).split())





def print_table(header_values, content, colors):

    data = go.Table(

    

      header = dict(

        values = header_values ,

        line = dict(color = "rgb(70,130,180)"),

        fill = dict(color = "rgb(70,130,180)"),

        align = 'center',

        font = dict(color = 'black', size = 12)

      ),

      cells = dict(

        values = content,

        fill = colors,  

        align = 'center',

        font = dict(color = 'black', size = 9),

        height = 40

        ))



    plotly.offline.iplot([data])
metadata["filename"] = metadata["handle"].apply(lambda x: x[1:])

metadata["age"] = ((datetime.today() - pd.to_datetime(metadata["born"])).dt.days/365).astype(int)

dems = (metadata[metadata["party"] == "D"]).copy()



dic = dict()

for index, row in metadata.iterrows():

    

    dic[row["filename"]] = pd.read_csv("../input/twitter/%s.csv"%row["filename"])

    

    df = dic[row["filename"]]



    df["clean tweet"] = df['Text'].apply(clean_tweet)

    

    df = df[df["Language"] == "English"]

    metadata.loc[index,"first tweet in dataset"] = df["Created At"].astype("datetime64").min()

    metadata.loc[index,"number of tweets and retweets in dataset"] = df.shape[0]

    metadata.loc[index,"number of tweets in dataset"] = df[df["Tweet Type"] == "Tweet"].shape[0]

    metadata.loc[index,"average likes all time"] = int(df[df["Tweet Type"] == "Tweet"]["Favorites"].mean()+0.5)

    metadata.loc[index,"average retweets all time"] = int(df[df["Tweet Type"] == "Tweet"]["Retweets"].mean()+0.5)

    

    df = df[df["Created At"].astype("datetime64").dt.year > 2018]

    metadata.loc[index,"number of tweets in 2019"] = df[df["Tweet Type"] == "Tweet"].shape[0]

    metadata.loc[index,"average likes in 2019"] = int(df[df["Tweet Type"] == "Tweet"]["Favorites"].mean()+0.5)

    metadata.loc[index,"average retweets in 2019"] = int(df[df["Tweet Type"] == "Tweet"]["Retweets"].mean()+0.5)
columns =  ["name", "sex", "born", "age","announcement", "party", "city of residence", "state of residence", "children"]



header_values = ['<b>%s</b>'%x for x in columns]

content = metadata[columns].T

colors = dict()

print_table(header_values, content, colors)
sns.distplot(dems["age"], bins=6, kde=False, rug=True, color="#3498db" )
children_counts = dems["children"].value_counts()

sns.barplot(children_counts.index, children_counts.values, color="#3498db")
states_count = dems['state of residence'].value_counts()



text = []

for state in states_count.index:

    text.append("<br>".join(dems[dems['state of residence'] == state]["name"]))

    

scl = [

    [0.0, 'rgb(174, 214, 241)'],

    [0.5, 'rgb(52, 152, 219)'],

    [1.0, 'rgb(33, 97, 140)'],

]

    



    



data = [go.Choropleth(

    colorscale = scl,

    autocolorscale = False,

    locations = states_count.index,

    z = states_count,

    locationmode = 'USA-states',

    text = text,

    marker = go.choropleth.Marker(

        line = go.choropleth.marker.Line(

            color = 'rgb(0,0,0)',

            width = 1

        )),

    colorbar = go.choropleth.ColorBar(

        title = "")

)]



layout = go.Layout(

    title = go.layout.Title(

        text = 'Where are the Democratic Candidates coming from (mouseover for candidate names)'

    ),

    geo = go.layout.Geo(

        scope = 'usa',

        projection = go.layout.geo.Projection(type = 'albers usa'),

        showlakes = True,

        lakecolor = 'rgb(255, 255, 255)'),

)



fig = go.Figure(data = data, layout = layout)

plotly.offline.iplot(fig)
sorted_data = metadata.sort_values("average likes in 2019", ascending=False)

col = ["name", "average likes all time", "average likes in 2019",

               "average retweets all time", "average retweets in 2019"]

tidy = sorted_data[col].melt(id_vars='name')





f, ax = plt.subplots(figsize=(10, 16))

sns.barplot(x='value', y='name', hue='variable', data = tidy, orient = "h")
#pair_of_words_list =[]

words_list =[]



for index, row in metadata.iterrows():

    df = dic[row["filename"]]

    df = df[df["Tweet Type"] == "Tweet"]

    df = df[df["Created At"].astype("datetime64").dt.year > 2018]

    df = df[df["Language"] == "English"]

    words = ' '.join(df["clean tweet"]).split(" ")

    words_list.extend(words)

#    pair_of_words = []

#    for i in range(len(words)-1):

#        pair_of_words.append(words[i]+ " " + words[i+1])

#    pair_of_words_list.extend(pair_of_words)

#pair_of_words_list = pd.Series(pair_of_words_list)



common_words = pd.Series(words_list).value_counts()[:100].index.tolist()

add_to_common = []

common_words.extend(add_to_common)

exceptions = ["first", "right", "campaign", "work", 'people','country','president','we_are','trump','great','we_need','thank_you','vote','fight','america','women','state','thanks','help','support','americans']



common_words = [x for x in common_words if x not in exceptions]



buzz_words = pd.DataFrame()

buzz_value = pd.DataFrame()



for index, row in metadata.iterrows():



    df = dic[row["filename"]]

    df = df[df["Created At"].astype("datetime64").dt.year > 2018]

    df = df[df["Tweet Type"] == "Tweet"]

    df = df[df["Language"] == "English"]

    name = row["name"]

    words = pd.Series(' '.join(df["clean tweet"]).split(" "))

    words = (words[~words.isin(common_words)])

    words_count = words.value_counts()*100/len(words)

    if "women" in words_count.index:

        metadata.loc[index,"women mention"] = int(words_count["women"]*1000+0.5)/1000

    else:

        metadata.loc[index,"women mention"] = 0

    if "country" in words_count.index:

        metadata.loc[index,"country mention"] = int(words_count["country"]*1000+0.5)/1000

    else:

        metadata.loc[index,"country mention"] = 0

    buzz_words[row["name"]] = [x[0] + "<br>" + str(int(x[1]*10+0.5)/10) for x in zip(words_count[:10].index, words_count[:10].values)]

    buzz_value[row["name"]] = words_count[:10].values



colors = cl.scales['9']['seq']['YlOrRd']

buzz_value = (buzz_value / buzz_value.max().max()*9-0.01).astype(int)

buzz_value = buzz_value.applymap(lambda x: colors[x])



header_values = ['<b>Name</b>']

content = np.concatenate((np.expand_dims(buzz_words.columns,0), buzz_words))

colors = dict(color = np.concatenate((np.expand_dims(["rgb(135,206,235)"]* buzz_value.shape[1],0), buzz_value)))

print_table(header_values, content, colors)
sorted_data = metadata.sort_values("women mention", ascending=False)

col = ["name", "women mention", "country mention"]

tidy = sorted_data[col].melt(id_vars='name')



palette = ['#e74c3c','#3498db']



f, ax = plt.subplots(figsize=(8, 10))

sns.barplot(x='value', y='name', hue='variable', palette=palette, data = tidy, orient = "h")
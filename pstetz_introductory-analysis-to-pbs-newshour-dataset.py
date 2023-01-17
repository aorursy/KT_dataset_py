### Standard imports
import pandas as pd
import numpy as np
pd.options.display.max_columns = 50

### Time imports
import datetime
import time

# Counter
from collections import Counter

# Operator
import operator

# Regular Expressions
import re

# Directory helper
import glob

# Language processing import
import nltk

# Random
import random

# Progress bar
from tqdm import tqdm

### Removes warnings that occassionally show in imports
import warnings
warnings.filterwarnings('ignore')
### Standard imports
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()

### Altair
import altair as alt
alt.renderers.enable('notebook')

### Plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
init_notebook_mode(connected=True)

# WordCloud
from wordcloud import WordCloud

# Folium
import folium
# A short hand way to plot most bar graphs
def pretty_bar(data, ax, xlabel=None, ylabel=None, title=None, int_text=False, x=None, y=None):
    
    if x is None:
        x = data.values
    if y is None:
        y = data.index
    
    # Plots the data
    fig = sns.barplot(x, y, ax=ax)
    
    # Places text for each value in data
    for i, v in enumerate(x):
        
        # Decides whether the text should be rounded or left as floats
        if int_text:
            ax.text(0, i, int(v), color='k', fontsize=14)
        else:
            ax.text(0, i, round(v, 3), color='k', fontsize=14)
     
    ### Labels plot
    ylabel != None and fig.set(ylabel=ylabel)
    xlabel != None and fig.set(xlabel=xlabel)
    title != None and fig.set(title=title)

def pretty_transcript(transcript, limit_output=0):
    for i, speaker in enumerate(transcript):
        if limit_output and i > limit_output:
            print("  (...)")
            break
        print(color.UNDERLINE, speaker[0] + ":", color.END)
        for txt in speaker[1:]:
            print("\n\n   ".join(txt))
        print()
    
def get_trend(series, ROLLING_WINDOW=16):
    trend = series.rolling(
        window=ROLLING_WINDOW,
        center=True, min_periods=1).mean()

    trend = trend.rolling(
        window=ROLLING_WINDOW // 2,
        center=True, min_periods=1).mean()

    trend = trend.rolling(
        window=ROLLING_WINDOW // 4,
        center=True, min_periods=1).mean()
    return trend
    
### Used to style Python print statements
class color:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
pbs = pd.read_json("../input/PBS-newhour-clean.json")
pbs = pbs.sort_values("Date")

pbs.Story.fillna("", inplace=True)

pbs["Year"]  = pbs.Date.map(lambda x: x.year)
pbs["Month"] = pbs.Date.map(lambda x: x.month)

print("Shape of pbs:", pbs.shape)
pbs.head()
pbs.Timezone.value_counts()
temp = pbs.iloc[0]

print(temp.Title)
print(temp.URL)
temp = pbs[pbs.Transcript.map(lambda x: x != [])].iloc[0]

print(f"{color.BOLD}{temp.Date}{color.END}")
print(f"{color.BOLD}{temp.Title}{color.END}")
print()
pretty_transcript(temp.Transcript, limit_output=2)
for i in range(5):
    print(pbs.iloc[i].Date)
    print(pbs.iloc[i].Story)
    print()
temp = (pbs
        .assign(n=0)
        .set_index("Date")
        .groupby(pd.Grouper(freq="M"))
        .n
        .apply(len)
        .sort_index()
)

trace = go.Scatter(
        x=temp.index,
        y=temp.values,
    )

layout = go.Layout(
    title = "Number of transcripts available over time",
    yaxis=dict(title="Number of transcripts"),
    xaxis=dict(title="Date"),
)



fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
### These are just examples
pois = {0: "BERNIE SANDERS",
        1: "VLADIMIR PUTIN",
        2: "DONALD TRUMP",
        3: "JUDY WOODRUFF",
        4: "BEN CARSON",
        5: "STEPHEN COLBERT",
        6: "HILLARY CLINTON",
        7: "JOHN F. KENNEDY",
        8: "ANGELA MERKEL",
        9: "JEFF BEZOS",
        10: "XI JINPING"
}

poi = pois[2]

print("Showing results for:", poi)
pbs[pbs.Speakers.map(lambda x: poi in x)].head(3)
pois = ["BERNIE SANDERS", "DONALD TRUMP", "HILLARY CLINTON",
        "BARACK OBAMA", "MITT ROMNEY", "ANGELA MERKEL",
        "JOSEPH BIDEN", "MIKE PENCE"]

def get_num_articles(df, poi):
    num_articles = len(df[df.Speakers.map(lambda x: poi in x)])
    return num_articles

def get_num_words(df, poi):
    speaker_text = list()
    transcripts  = df[df.Speakers.map(lambda x: poi in x)].Transcript.values
    num_words    = 0
    
    for transcript in transcripts:
        for person in transcript:
            if person[0] == poi:
                for txt in person[1]:
                    num_words += len(txt.split(" "))
    return num_words

articles, words = list(), list()

for poi in pois:
    num_articles = get_num_articles(pbs, poi)
    num_words    = get_num_words(pbs, poi)
    
    articles.append(num_articles)
    words.append(num_words)

trace1 = go.Bar(
    x=pois,
    y=articles,
    name='Total articles'
)
trace2 = go.Bar(
    x=pois,
    y=words,
    name='Total words'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
iplot(fig);
persons = pbs.Speakers.map(list).sum()

freq = sorted(Counter(persons).items(), key=operator.itemgetter(1), reverse=True)
x, y = list(zip(*freq[:25]))

fig, ax = plt.subplots(1, 1, figsize=(14, 14))
temp = pd.Series(list(y), index=list(x))
pretty_bar(temp, ax, title="Top Speakers", xlabel="Number of Transcripts");
LIMIT_TIME = True
topics     = ["Obama", "Trump", "Clinton", "Bush", "Immigration", "Congress", "Racism"]

def topic_popularity(topic):
    def popularity_helper(transcript):
        transcript = list(map(lambda x: x[1][0], transcript))
        transcript = (" ".join(transcript).lower()).split(" ")
        N          = len(transcript)
        counts     = Counter(transcript)
        return (counts[topic.lower()] / N) * 100
    return popularity_helper

if LIMIT_TIME:
    temp = pbs[pbs.Year > 2010]
else:
    temp = pbs

datas = []
for topic in tqdm(topics):
    temp["Temp"] = (
                temp[temp.Transcript.map(lambda x: x != [])]
                    .Transcript
                    .map(topic_popularity(topic))
                )

    data = (temp
         .set_index("Date")
         .groupby(pd.Grouper(freq="M"))
         .Temp
         .apply(np.mean)
    )

    trend = get_trend(data, ROLLING_WINDOW=12)

    datas.append((topic, data, trend))

traces = []

for topic, data, _ in datas:
    traces.append(go.Scatter(
                            x=data.index,
                            y=data.values,
                            name=f"{topic} - actual"
                        ))
    
for topic, _, trend in datas:
    traces.append(go.Scatter(
                            x=trend.index,
                            y=trend.values, 
                            name=f"{topic} - trend"
                        ))
buttons = []

for i, topic in enumerate(topics):
    visibility = [i==j for j in range(len(topics))]
    button = dict(
                 label =  topic,
                 method = 'update',
                 args = [{'visible': visibility},
                     {'title': f"'{topic}' usage over time" }])
    buttons.append(button)

updatemenus = list([
    dict(active=-1,
         x=-0.15,
         buttons=buttons
    )
])

layout = dict(title='Topic popularity', 
              updatemenus=updatemenus,
                xaxis=dict(title='Date'),
                yaxis=dict(title='Percent of words')
             )

fig = dict(data=traces, layout=layout)
fig['layout'].update(height=800, width=800)

iplot(fig)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

Tweets = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv")
import numpy as np

import plotly.graph_objects as go

import colorlover as cl
Tweets.info()
Tweets.sample(3)
Tweets.airline.value_counts()
Tweets.airline_sentiment.value_counts()
Tweets[Tweets.airline == 'American'].airline_sentiment.value_counts()
category_order = [

    'negative',

    'neutral',

    'positive'

]



# rearrange the data into the format we desire to only show airline and its sentiment proportions

Tweets_airline = pd.pivot_table(

    Tweets,

    index = 'airline',

    columns = 'airline_sentiment',

    values = 'tweet_id',

    aggfunc = 'count'

)



# reorder the columns as desired above

Tweets_airline = Tweets_airline[category_order]



# make specific columns to represent undesired or negative answers

Tweets_airline.negative = Tweets_airline.negative * -1
Tweets_airline
# sort by desired column

Tweets_airline = Tweets_airline.sort_values(by='negative', ascending = False)



fig = go.Figure()



for column in Tweets_airline.columns:

    fig.add_trace(go.Bar(

        x = Tweets_airline[column],

        y = Tweets_airline.index,

        name = column,

        orientation = 'h',

        marker_color = cl.scales[str(len(category_order))]['div']['RdYlGn'][category_order.index(column)],

    ))



fig.update_layout(

    barmode = 'relative',

    title = 'Twitter Sentiment Analysis of US Airlines'

)

fig.show()
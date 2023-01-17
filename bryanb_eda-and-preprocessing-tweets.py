# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
from textblob import TextBlob
from pandas.io.json import json_normalize
from wordcloud import WordCloud
import math
import re
import json
# Preprocessing functions

def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
def analyze_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
    
def convert_month_to_number(month):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months.index(month)+1

def set_timestamp(df):
    dates = pd.DataFrame(columns = ['Year', 'Month', 'Day'])
    dates['Year'] = df['Date'].apply(lambda x: int(x[-4:]))
    dates['Month'] = df['Date'].apply(lambda x: x[4:7]).apply(lambda x:int(convert_month_to_number(x)))
    dates['Day'] = df['Date'].apply(lambda x: int(x[8:10].rsplit()[0]))

    df.Date = pd.to_datetime(dates)
    return df
column_names = ['Date', 'Text', 'Likes', 'Retweets', 'Sentiment', 'Source', 'Length']

# Create empty dataframe with column names
valorant_df = pd.DataFrame(columns = column_names)

with open('../input/valorant-tweets/tweets.txt') as f:
    for line in f:
        if len(line) > 10: # Dont take into account empty lines
            to_append = {}
            obs = json.loads(line)
            
            to_append['Date'] = obs['created_at']
            to_append['Text'] = obs['text']
            to_append['Likes'] = obs['favorite_count'] 
            to_append['Retweets'] = obs['retweet_count']
            
            valorant_df = valorant_df.append(to_append, ignore_index = True)

valorant_df = set_timestamp(valorant_df)
valorant_df['Sentiment'] = valorant_df['Text'].apply(lambda x:analyze_sentiment(x))
valorant_df['Source'] = 'random_user'
valorant_df['Length'] = valorant_df['Text'].apply(len)
valorant_df['Word_counts'] = valorant_df['Text'].apply(lambda x:len(str(x).split()))
valorant_df.head()
print(f'{valorant_df.shape[0]} observations, {valorant_df.shape[1]} columns')
# Find emoji patterns
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

# Basic function to clean the text
def clean_text(text):
    text = str(text)
    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    # Remove identifications
    text = re.sub(r'@\w+', '', text)
    # Remove links
    text = re.sub(r'http.?://[^/s]+[/s]?', '', text)
    return text.strip().lower()
valorant_df['Text'] = valorant_df['Text'].apply(lambda x:clean_text(x))
# Count of observations per category
valorant_df['Sentiment'].value_counts(normalize = True)
x = ['Neutral', 'Positive', 'Negative']
y = [6959, 3177, 1338]

# Use the hovertext kw argument for hover text
fig = go.Figure(data=[go.Bar(x=x, y=y,
            hovertext=['61% of tweets', '28% of tweets', '11% of tweets'])])

# Customize aspect
#marker_color='rgb(158,202,225)'
fig.update_traces(marker_line_color='midnightblue',
                  marker_line_width=1.)
fig.update_layout(title_text='Distribution of sentiment')
fig.show()
neutral = valorant_df[valorant_df['Sentiment'] == 0]
positive = valorant_df[valorant_df['Sentiment'] == 1]
negative = valorant_df[valorant_df['Sentiment'] == -1]
#neutral_text
print("Neutral tweet example  :",neutral['Text'].values[1])
# Positive tweet
print("Positive Tweet example :",positive['Text'].values[1])
#negative_text
print("Negative Tweet example :",negative['Text'].values[1])
x = valorant_df.Length.values
#x = [math.log10(i) for i in list(valorant_df.Length.values) if i!= 0]

fig = go.Figure(data=[go.Histogram(x=x,
                                   marker_line_width=1, 
                                   marker_line_color="midnightblue", 
                                   xbins_size = 5)])

fig.update_layout(title_text='Distribution of tweet lengths')
fig.show()
x1 = neutral.Length.values
x2 = positive.Length.values
x3 = negative.Length.values

fig = go.Figure(data=[go.Histogram(x=x1,
                                   marker_line_width=1, 
                                   marker_line_color="midnightblue", 
                                   xbins_size = 5,
                                   opacity = 0.5)])

fig.update_layout(title_text='Distribution of neutral tweet lengths')
fig.show()

fig = go.Figure(data=[go.Histogram(x=x2,
                                   marker_line_width=1, 
                                   marker_color='rgb(50,202,50)', 
                                   marker_line_color="midnightblue", 
                                   xbins_size = 5,
                                   opacity = 0.5)])

fig.update_layout(title_text='Distribution of positive tweet lengths')
fig.show()

fig = go.Figure(data=[go.Histogram(x=x3,
                                   marker_line_width=1, 
                                   marker_color='crimson', 
                                   marker_line_color="midnightblue", 
                                   opacity = 0.5)])

fig.update_layout(title_text='Distribution of negative tweet lengths')
fig.show()
y1 = neutral.Length.values
y2 = positive.Length.values
y3 = negative.Length.values

fig = go.Figure()

fig.add_trace(go.Box(y=y1, 
                     name="Neutral", 
                     marker_line_width=1, 
                     marker_line_color="midnightblue"))

fig.add_trace(go.Box(y=y2, 
                     name="Positive", 
                     marker_line_width=1, 
                     marker_color = 'rgb(50,202,50)'))

fig.add_trace(go.Box(y=y3, 
                     name="Negative", 
                     marker_line_width=1, 
                     marker_color = 'crimson'))

fig.update_layout(title_text="Box Plot tweet lengths")

fig.show()
x = valorant_df.Word_counts.values
#x = [math.log10(i) for i in list(valorant_df.Length.values) if i!= 0]

fig = go.Figure(data=[go.Histogram(x=x,
                                   marker_line_width=1, 
                                   marker_line_color="midnightblue")])

fig.update_layout(title_text='Distribution of tweet lengths')
fig.show()
x1 = neutral.Word_counts.values
x2 = positive.Word_counts.values
x3 = negative.Word_counts.values

fig = go.Figure(data=[go.Histogram(x=x1,
                                   marker_line_width=1, 
                                   marker_line_color="midnightblue", 
                                   opacity = 0.5)])

fig.update_layout(title_text='Distribution of neutral tweet lengths')
fig.show()

fig = go.Figure(data=[go.Histogram(x=x2,
                                   marker_line_width=1, 
                                   marker_color='rgb(50,202,50)', 
                                   marker_line_color="midnightblue", 
                                   opacity = 0.5)])

fig.update_layout(title_text='Distribution of positive tweet lengths')
fig.show()

fig = go.Figure(data=[go.Histogram(x=x3,
                                   marker_line_width=1, 
                                   marker_color='crimson', 
                                   marker_line_color="midnightblue", 
                                   opacity = 0.5)])

fig.update_layout(title_text='Distribution of negative tweet lengths')
fig.show()
y1 = neutral.Word_counts.values
y2 = positive.Word_counts.values
y3 = negative.Word_counts.values

fig = go.Figure()

fig.add_trace(go.Box(y=y1, 
                     name="Neutral", 
                     marker_line_width=1, 
                     marker_line_color="midnightblue"))

fig.add_trace(go.Box(y=y2, 
                     name="Positive", 
                     marker_line_width=1, 
                     marker_color = 'rgb(50,202,50)'))

fig.add_trace(go.Box(y=y3, 
                     name="Negative", 
                     marker_line_width=1, 
                     marker_color = 'crimson'))

fig.update_layout(title_text="Box Plot word counts")

fig.show()
def wordcloud(df, text = 'Text'):
    
    # Join all tweets in one string
    corpus = " ".join(str(review) for review in df[text])
    
    print (f"There are {len(corpus)} words in the combination of all review.")
    
    wordcloud = WordCloud(max_font_size=50, 
                          max_words=100, 
                          collocations=False,
                          background_color="white").generate(corpus)
    
    plt.figure(figsize=(15,15))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
print('Neutral Wordcloud')
wordcloud(df = neutral)

print('Positive Wordcloud')
wordcloud(df = positive)

print('Negative Wordcloud')
wordcloud(df = negative)
# This dataset contains all tweets from official Valorant accounts
from_RIOT = pd.read_csv('../input/valorant-tweets/Valorant.csv', usecols = ['Date', 'Text', 'Likes', 'Retweets', 'Sentiment', 'Source', 'Length'])
from_RIOT = from_RIOT.reindex(columns=['Date', 'Text', 'Likes', 'Retweets', 'Sentiment', 'Source', 'Length'])
from_RIOT['Date'] = pd.to_datetime(from_RIOT['Date'], format ='%Y/%m/%d')
from_RIOT['Word_counts'] = from_RIOT['Text'].apply(lambda x:len(str(x).split()))
# Merge the dataframes is possible has we have the source of accounts each tweet comes from
dataframe = pd.concat([valorant_df, from_RIOT])
dataframe.to_csv('all_tweets.csv')
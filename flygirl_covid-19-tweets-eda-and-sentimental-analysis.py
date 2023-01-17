import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pycountry
import warnings
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.dates as mdates
import re
import string
from collections import Counter
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import shuffle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.regularizers import l2

np.random_seed = 0


warnings.filterwarnings('ignore')
df = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
df.head()
print("Total number of records in data: ",len(df))
df.info()
df['user_location'].fillna('unknown', inplace=True)
df['user_location'].value_counts()
c = list(pycountry.countries)
def correct_location(x):
    for i in c:
        if str(i.name).lower() in x or str(i.alpha_2).lower() in x.split() or str(i.alpha_3).lower() in x.split():
            return str(i.name)
    return x
df['user_location'] = df['user_location'].apply(lambda x: correct_location(x.lower()))
df['user_location'].value_counts()
def plot_bar(x,y,title,x_label,y_label):
    fig = go.Figure(data=[go.Bar(
                x=x,
                y=y,
                text=y,
                textposition='auto',
            )])

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label
    )

    fig.show()
plot_bar(df['user_location'].value_counts().index[0:20],
         df['user_location'].value_counts().values[0:20],
         "Top 20 Locations by the number of tweets",
         "Location",
         "Tweet Count")
def plot_time_series(dates, counts, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Configure x-ticks
    ax.set_xticks(dates) # Tickmark + label at every plotted point
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
   
    ax.plot_date(dates, counts, ls='-', marker='o')
    ax.set_title(title)
    ax.set_ylabel('Tweet Count)')
    ax.grid(True)

    # Format the x-axis for dates (label formatting, rotation)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    fig.show()
df['just_date'] = pd.to_datetime(df['date']).dt.normalize()

top = list(df['user_location'].value_counts().index[1:5])
df_top = df[df['user_location'].isin(top)]
grp_df = df_top.groupby('user_location')
for i, grp in grp_df:
    dates = grp['just_date'].value_counts().sort_index().index
    counts = grp['just_date'].value_counts().sort_index().values
        
    plot_time_series(dates,counts,'Tweet count trend for '+i)
plot_bar(df['user_name'].value_counts().index[0:20],
         df['user_name'].value_counts().values[0:20],
         "Top 20 Users by the number of tweets",
         "User",
         "Tweet Count")
plot_bar(df['source'].value_counts().index[0:20],
         df['source'].value_counts().values[0:20],
         "Top 20 Tweeter Sources by the number of tweets",
         "Source",
         "Tweet Count")
stopwords_ = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        collocations=False,
        background_color='white',
        stopwords=stopwords_,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(df['text'], "WordCloud for tweets")
hashtags = list(df['hashtags'].dropna())

hashtags = [x.replace("'", '') for x in hashtags]
hashtags = [(re.sub(r'[^\w\s]','',x)).lower() for x in hashtags]

show_wordcloud(' '.join(hashtags),"WordCloud for hashtags")
data = pd.read_csv('../input/twitterdata/finalSentimentdata2.csv', encoding='"ISO-8859-1"')
data.drop(columns=['Unnamed: 0'], inplace=True)
data.head()
data['sentiment'].value_counts()
def clean_text(text):
    
    #remove urls
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)
    
    #remove html
    html_pattern = re.compile(r'<.*?>')
    text = html_pattern.sub(r'', text)
    
    #remove emojis
    emoji_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    text = emoji_pattern.sub(r'',text)
    
    #remove punctuations
    table = str.maketrans("", "", string.punctuation)
    text = text.translate(table)

    return text
#     #remove stopwords
#     stop = set(stopwords.words('english'))
#     text = [word.lower() for word in text.split() if word.lower() not in stop]

#     return ' '.join(text)
data['sentiment'] = data['sentiment'].apply(lambda x: int(x == 'joy'))
data['text'] = data['text'].apply(lambda x: clean_text(x))
df['text'] = df['text'].apply(lambda x: clean_text(x))
def word_counter(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count  

counter = word_counter(data['text'].append(df['text']))
num_of_words = len(counter)
max_len = 20

t = Tokenizer(num_words = num_of_words)
t.fit_on_texts(data['text'].append(df['text']))
train_x, test_x, train_y, test_y = train_test_split(data['text'], data['sentiment'], test_size=0.1, random_state=30)
train_tweets = t.texts_to_sequences(train_x)
train_tweets_padded = pad_sequences(train_tweets, maxlen=max_len, padding='post', truncating='post')

test_tweets = t.texts_to_sequences(test_x)
test_tweets_padded = pad_sequences(test_tweets, maxlen=max_len, padding='post', truncating='post')
model = Sequential()
model.add(Embedding(num_of_words, 30, input_length = max_len))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
data['sentiment'].value_counts()
weights = {0:1, 1:2}
history = model.fit(train_tweets_padded, train_y, epochs=5, validation_data = (test_tweets_padded,test_y),class_weight=weights,batch_size=40)
pred_y = model.predict_classes(test_tweets_padded)
print("F1 Score : ", f1_score(test_y,pred_y))
original_tweet = t.texts_to_sequences([df['text'][8]])
tweet_padded = pad_sequences(original_tweet, maxlen=max_len, padding='post', truncating='post')
res = model.predict_classes(tweet_padded)
print("Tweet :",df['text'][8])
print("Sentiment :",res[0][0])
original_tweet = t.texts_to_sequences([df['text'][644]])
tweet_padded = pad_sequences(original_tweet, maxlen=max_len, padding='post', truncating='post')
res = model.predict_classes(tweet_padded)
print("Tweet :",df['text'][644])
print("Sentiment :",res[0][0])
original_tweets = t.texts_to_sequences(df['text'])
tweets_padded = pad_sequences(original_tweets, maxlen=max_len, padding='post', truncating='post')
res = model.predict_classes(tweets_padded)
res = [i[0] for i in res]
plot_bar(list(Counter(res).keys()),
         list(Counter(res).values()),
         "Count of Positive(1) and Negative(0) sentiments",
         "Sentiment",
         "Tweet Count")

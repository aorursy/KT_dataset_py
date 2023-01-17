# importing dependency libraries

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.io.json import json_normalize

import  nltk, urllib.error, urllib.parse, urllib.request, json, datetime



outlets = 'abc-news,cnn,the-huffington-post,fox-news,usa-today,reuters,politico,the-washington-post,nbc-news,cbs-news,newsweek'



# generates the date one month ago to not break API request

def monthAgoDate():

    today = datetime.datetime.today()

    if today.month == 1:

        one_month_ago = today.replace(year=today.year - 1, month=12)

    else:

        extra_days = 0

        while True:

            try:

                one_month_ago = today.replace(month=today.month - 1, day=today.day - extra_days)

                break

            except ValueError:

                extra_days += 1

    return one_month_ago



# API keys

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("newsapi")



# function formatting url request using the 'qInTitle' function to only look at headlines

def getStories(query,date,sources):

        key = secret_value_0

        url = "http://newsapi.org/v2/everything?qInTitle="+query+"&sources="+sources+"&pageSize=100&apiKey="+key

        return urllib.request.urlopen(url)
# retrieving stories about each candidate

warren_stories = json.load(getStories('elizabeth+warren',monthAgoDate(), outlets))

biden_stories = json.load(getStories('joe+biden', monthAgoDate(), outlets))

bernie_stories = json.load(getStories('bernie+sanders', monthAgoDate(), outlets))

pete_stories = json.load(getStories('pete+buttigieg', monthAgoDate(), outlets))

amy_stories = json.load(getStories('amy+klobuchar', monthAgoDate(), outlets))

bloomberg_stories = json.load(getStories('mike+bloomberg', monthAgoDate(), outlets))

gabbard_stories = json.load(getStories('tulsi+gabbard', monthAgoDate(), outlets))
# creating empty data frames in memory

df_ew = pd.DataFrame()

df_jb = pd.DataFrame()

df_bs = pd.DataFrame()

df_pb = pd.DataFrame()

df_ak = pd.DataFrame()

df_mb = pd.DataFrame()

df_tg = pd.DataFrame()

    

# adding data to each frame, reducing the columns to essential, removing NaN values, and adding a column for the candidate's name.

def serializeFrames(dataframe, data, candidate):

    dataframe = json_normalize(data['articles'])

    dataframe = dataframe[['title','description','publishedAt','source.name']]

    dataframe = dataframe.dropna()

    dataframe['candidate'] = candidate

    return dataframe



# adding the data - probably could have handled this with a loop

df_ew = serializeFrames(df_ew, warren_stories, 'Warren')

df_jb = serializeFrames(df_jb, biden_stories, 'Biden')

df_bs = serializeFrames(df_bs, bernie_stories, 'Sanders')

df_pb = serializeFrames(df_pb, pete_stories, 'Buttigieg')

df_ak = serializeFrames(df_ak, amy_stories, 'Klobuchar')

df_mb = serializeFrames(df_mb, bloomberg_stories, 'Bloomberg')

df_tg = serializeFrames(df_tg, gabbard_stories, 'Gabbard')



# adding all of this data into one frame to rule them all

df = pd.concat([df_ew, df_jb, df_bs, df_pb, df_ak, df_mb, df_tg], ignore_index=True)

df.head()
# Sentiment analysis using VADER from Natural Language Tool Kit.

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()



# apply analysis and score results. 0.05 threshold currently being used.

def analyze_sent(sentence):

    snt = sia.polarity_scores(sentence)

    if snt['compound'] >= 0.05:

        return "Positive"

    elif snt['compound'] <= -0.05:

        return "Negative"

    else:

        return "Neutral"

    

analyze_sent('I am so happy')



def apply_sent_analysis(data):

    data['title_sentiment'] = data['title'].apply(lambda sentence: analyze_sent(sentence))

    data['description_sentiment'] = data['description'].apply(lambda sentence: analyze_sent(sentence))



apply_sent_analysis(df)

df.head()
import plotly.express as px

import plotly.graph_objects as go 

from plotly.subplots import make_subplots



all_headlines = px.histogram(df, x='title_sentiment', 

                             color='title_sentiment', 

                             title='Headline Sentiment for All Candidates', 

                             labels={'title_sentiment':'Sentiment'})

all_headlines.show()
all_outlets = px.histogram(df, x='source.name', 

                             color='title_sentiment', 

                             title='Headline Sentiment by Outlet for All Candidates', 

                             labels={'title_sentiment':'Sentiment', 'source.name':'News Outlet'})

all_outlets.show()
all_sentiment = px.histogram(df, x='candidate', 

                             color='title_sentiment', 

                             title='Headline Sentiment by candidate')

all_sentiment.show()
all_lede_sentiment = px.histogram(df, x='candidate', 

                             color='description_sentiment', 

                             title='Lede Sentiment by candidate')

all_lede_sentiment.show()
df["title_sentiment"] = df["title_sentiment"].astype("category")

df['description_sentiment'] = df['description_sentiment'].astype("category")
#pivot title sentiments to get counts of pos, neu, neg headlines

df_title_sent = df.groupby(['candidate', 'title_sentiment'])[['title_sentiment']].count()



#unstack the chaos done by pivoting: one layer of column headers and numeric index

df_title_sent = df_title_sent.unstack()

df_title_sent.columns = [' '.join(col).strip() for col in df_title_sent.columns.values]

df_title_sent = df_title_sent.reset_index(level='candidate')



#melting it back into long format

df_title_sent = df_title_sent.melt(id_vars=['candidate'])
# Doing the same thing as above with description/lede sentiments

df_desc_sent = df.groupby(['candidate', 'description_sentiment'])[['description_sentiment']].count()

df_desc_sent = df_desc_sent.unstack()

df_desc_sent.columns = [' '.join(col).strip() for col in df_desc_sent.columns.values]

df_desc_sent = df_desc_sent.reset_index(level='candidate')



df_desc_sent = df_desc_sent.melt(id_vars=['candidate'])
# putting the pieces back together!

df_sent_count = pd.concat([df_title_sent, df_desc_sent], ignore_index=True)

df_sent_count.head()
df2 = pd.concat([df_ew, df_jb, df_bs, df_pb, df_ak, df_mb, df_tg], ignore_index=True)



def sent_compound(sentence):

    snt = sia.polarity_scores(sentence)

    return snt['compound']



def apply_sent_2(data):

    data['title_sentiment'] = data['title'].apply(lambda sentence: sent_compound(sentence))

    data['description_sentiment'] = data['description'].apply(lambda sentence: sent_compound(sentence))



apply_sent_2(df2)

df2.head()
cand_x_headline = px.box(x=df2['candidate'], y=df2['title_sentiment'])

cand_x_headline.show()
cand_x_lede = px.box(x=df2['candidate'], y=df2['description_sentiment'])

cand_x_lede.show()
import scipy.stats as stats



df2_small = df2[['candidate','title_sentiment','description_sentiment']]



df2_ew = df2_small[df2_small.candidate == 'Warren']

df2_jb = df2_small[df2_small.candidate == 'Biden']

df2_bs = df2_small[df2_small.candidate == 'Sanders']

df2_pb = df2_small[df2_small.candidate == 'Buttigieg']

df2_ak = df2_small[df2_small.candidate == 'Klobuchar']

df2_mb = df2_small[df2_small.candidate == 'Bloomberg']

df2_tg = df2_small[df2_small.candidate == 'Gabbard']



stats.f_oneway(df2_ew['title_sentiment'],

               df2_jb['title_sentiment'],

               df2_bs['title_sentiment'],

               df2_pb['title_sentiment'],

               df2_ak['title_sentiment'],

               df2_mb['title_sentiment'],

               df2_tg['title_sentiment'],)
stats.f_oneway(df2_ew['description_sentiment'],

               df2_jb['description_sentiment'],

               df2_bs['description_sentiment'],

               df2_pb['description_sentiment'],

               df2_ak['description_sentiment'],

               df2_mb['description_sentiment'],

               df2_tg['description_sentiment'])
df_f = pd.concat([df2_ew, df2_ak, df2_tg], ignore_index=True)

df_m = pd.concat([df2_jb, df2_bs, df2_pb, df2_mb], ignore_index=True)



print("Male/Female comparison by headline: " + str(stats.ttest_ind(df_f['title_sentiment'],df_m['title_sentiment'], equal_var = False)))

print("Male/Female comparison by lede: " + str(stats.ttest_ind(df_f['description_sentiment'],df_m['description_sentiment'], equal_var = False)))
print(sent_compound("I'm so excited!"))

print(sent_compound("I'm so excited, I could die!"))
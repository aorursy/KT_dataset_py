####TWITTER API####
#### Wrangling Cleaning Preprocessing ####
#import pandas

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import emoji

import re

from bs4 import BeautifulSoup



# Set iPython's max column width to 1000

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_colwidth', -1)

pd.options.display.float_format = '{:,.0f}'.format


# clean_cny_text = pd.read_csv("../input/clean_cny_text.csv")

# r_emotion_cny = pd.read_csv("../input/r_emotion_cny.csv")

# r_sent_cny = pd.read_csv("../input/r_sent_cny.csv")



#load in json file

file = "../input/distweet/disresortcny.json"

df = pd.read_json(file,lines = True)
#select columns we want

df = df[['text','id',

          'lang','created_at',

          'user','source','retweeted_status',

          'extended_tweet', 'entities',

         'place', 'truncated', 'quoted_status','favorite_count',

         'reply_count', 'retweet_count', 'quote_count', 'in_reply_to_user_id']]
#Select only Japanese and English Text

df = df[(df['lang'] == 'ja') | (df['lang'] == 'en')]
#quoted status nested dictionary

df['quote_created_at'] = [d.get('created_at') if type(d) == dict else np.nan

                        for d in df['quoted_status']]

df['quote_text'] =  [d.get('text') if type(d) == dict else np.nan

                 for d in df['quoted_status']]

df['quoted_retweet_count'] = [d.get('retweet_count') if type(d) == dict else np.nan

                           for d in df['quoted_status']]

df['quoted_favorite_count'] = [d.get('favorite_count') if type(d) == dict else np.nan

                            for d in df['quoted_status']]

df['quoted_lang'] = [d.get('lang') if type(d) == dict else np.nan

                  for d in df['quoted_status']]
#retweeted status nested dictionary

df['rt_created_at'] = [d.get('created_at') if type(d) == dict else np.nan

                        for d in df['retweeted_status']]

df['rt_reply_count'] = [d.get('reply_count') if type(d) == dict else np.nan

                        for d in df['retweeted_status']]

df['rt_id'] =  [d.get('id') if type(d) == dict else np.nan

                 for d in df['retweeted_status']]

df['rt_text'] = [d.get('text') if type(d) == dict else np.nan

                  for d in df['retweeted_status']]

df['rt_source'] = [d.get('source') if type(d) == dict else np.nan

                    for d in df['retweeted_status']]

df['rt_user'] = [d.get('user') if type(d) == dict else np.nan

                  for d in df['retweeted_status']]

df['rt_retweet_count'] = [d.get('retweet_count') if type(d) == dict else np.nan

                           for d in df['retweeted_status']]

df['rt_favorite_count'] = [d.get('favorite_count') if type(d) == dict else np.nan

                            for d in df['retweeted_status']]

df['rt_lang'] = [d.get('lang') if type(d) == dict else np.nan

                  for d in df['retweeted_status']]



df['rt_extended_tweet'] = [d.get('extended_tweet') if type(d) == dict

                          else np.nan for d in df['retweeted_status']]



df['rt_full_text'] = [d.get('full_text') if type(d) == dict

                          else np.nan for d in df['rt_extended_tweet']]

df['rt_user_id'] = [d.get('id') if type(d) == dict

                          else np.nan for d in df['rt_user']]

df['rt_user_screen_name'] = [d.get('screen_name') if type(d) == dict

                          else np.nan for d in df['rt_user']]

df['rt_user_location'] = [d.get('location') if type(d) == dict

                          else np.nan for d in df['rt_user']]

df['rt_user_follower_count'] = [d.get('followers_count') if type(d) == dict

                          else np.nan for d in df['rt_user']]

df['rt_user_friends_count'] = [d.get('friends_count') if type(d) == dict

                          else np.nan for d in df['rt_user']]

df['rt_user_listed_count'] = [d.get('listed_count') if type(d) == dict

                          else np.nan for d in df['rt_user']]

df['rt_user_favorites_count'] = [d.get('favourites_count') if type(d) == dict

                          else np.nan for d in df['rt_user']]

df['rt_user_created_at'] = [d.get('created_at') if type(d) == dict

                          else np.nan for d in df['rt_user']]



df['rt_user_description'] = [d.get('extended_tweet') if type(d) == dict

                          else np.nan for d in df['rt_user']]
#pull out extended Tweets

df['ex_tw_full_text'] = [d.get('full_text') if type(d) == dict else np.nan

                          for d in df['extended_tweet']]
#selecting dictionaries from user, nested dictionary



df['user_id'] = [d.get('id') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_name'] = [d.get('name') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_screen_name'] = [d.get('screen_name') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_location'] = [d.get('location') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_description'] = [d.get('description') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_fol_count'] = [d.get('followers_count') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_fr_count'] = [d.get('friends_count') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_fav_count'] = [d.get('favourites_count') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_status_count'] = [d.get('statuses_count') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_created_at'] = [d.get('created_at') if type(d) == dict else np.nan

                          for d in df['user']]

df['user_listed_count'] = [d.get('listed_count') if type(d) == dict else np.nan

                          for d in df['user']]

#Clean up the source

df['source']= df['source'].replace(np.nan, '')

#df['source'] = [BeautifulSoup(text).get_text() if type(text) == dict else np.nan for text in df['source']]

df['source'] = [BeautifulSoup(text).get_text() for text in df['source']]

#clean rt_source

df['rt_source']= df['rt_source'].replace(np.nan, '')

df['rt_source'] = [BeautifulSoup(text).get_text() if text != np.nan else np.nan for text in df['rt_source']]
#change the date time column

df['created_at'] = pd.to_datetime(df['created_at'])

df['rt_created_at'] = pd.to_datetime(df['rt_created_at'])

df['quote_created_at'] = pd.to_datetime(df['quote_created_at'])

df['rt_user_created_at'] = pd.to_datetime(df['rt_user_created_at'])

df['user_created_at'] = pd.to_datetime(df['user_created_at'])
df = df.drop(columns = ['user', 'retweeted_status', 'entities', 'quoted_status', 'rt_extended_tweet', 'rt_user', 'extended_tweet',

                       'place'])
df.columns
df.loc[0:50, ['text', 'truncated', 'ex_tw_full_text','rt_text', 'rt_full_text']]
#combining text columns tweets

df['text'] = df['text'].replace(np.nan,'')

df['len_text'] = df['text'].apply(len)

df['combo_text'] = np.where(df['len_text'] < 140, df['text'], df['ex_tw_full_text'])

df['combo_text'] = df['combo_text'].replace(np.nan,'')



#combining text columns retweets

df['rt_text'] = df['rt_text'].replace(np.nan,'')

df['rt_len'] = df['rt_text'].apply(len)

df['rtcombo_text'] = np.where(df['rt_len'] < 140, df['rt_text'], df['rt_full_text'])

df['rtcombo_text'] = df['rtcombo_text'].replace(np.nan,'')



#combining retweets and tweets

df['rtcombo_len'] = df['rtcombo_text'].apply(len)

df['full_text'] = np.where(df['rtcombo_len'] == 0, df['combo_text'], df['rtcombo_text'])
df['full_text'][24:50]
####Translate Japanese Text to English export to excel for google translation then bring back as word document####
#Drop duplicate tweets that were retweeted during collection period 

df2 = df.drop_duplicates(subset = 'full_text')
df2['full_text']
df_ja = df2[df['lang'] == 'ja']

df_en = df2[df['lang'] == 'en']

#df_ja['full_text'].to_csv('cny_ja.csv')
#https://translate.google.com/#view=home&op=docs&sl=auto&tl=en
# import io

# import csv

# from docx import Document



# def read_docx_tables(filename, tab_id=None, **kwargs):

#     """

#     parse table(s) from a Word Document (.docx) into Pandas DataFrame(s)



#     Parameters:

#         filename:   file name of a Word Document



#         tab_id:     parse a single table with the index: [tab_id] (counting from 0).

#                     When [None] - return a list of DataFrames (parse all tables)



#         kwargs:     arguments to pass to `pd.read_csv()` function



#     Return: a single DataFrame if tab_id != None or a list of DataFrames otherwise

#     """

#     def read_docx_tab(tab, **kwargs):

#         vf = io.StringIO()

#         writer = csv.writer(vf)

#         for row in tab.rows:

#             writer.writerow(cell.text for cell in row.cells)

#         vf.seek(0)

#         return pd.read_csv(vf, **kwargs)



#     doc = Document(filename)

#     if tab_id is None:

#         return [read_docx_tab(tab, **kwargs) for tab in doc.tables]

#     else:

#         try:

#             return read_docx_tab(doc.tables[tab_id], **kwargs)

#         except IndexError:

#             print('Error: specified [tab_id]: {}  does not exist.'.format(tab_id))

#             raise
# translate = read_docx_tables('cny_goog_translate.docx', header = None)
# dftrans = pd.DataFrame(translate[0])

# dftrans.columns = ['index','trans_text']

# dftrans.set_index('index')
dftrans = pd.read_csv('../input/translation/dftrans.csv', index_col = 0)
df_ja = df_ja.reset_index(drop = True)

dftrans = dftrans.reset_index(drop = True)

df_en = df_en.reset_index(drop = True)

df_en['google_trans'] = 'None'
dftrans.columns = ['index', 'goog_trans']

df_ja['google_trans'] = dftrans['goog_trans']
df_ja.loc[:,['google_trans', 'full_text']]
#Selecting text column used for ML

df3 = pd.concat([df_en, df_ja]).sort_values('created_at')

df3 = df3.reset_index(drop = True)

df3 = df3.loc[:,['full_text','google_trans', 'text', 'rt_text','lang']]

df3['google_trans'] = df3['google_trans'].replace(np.nan, 'None')

df3['en_ja_goog'] = np.where((df3['google_trans'] == 'None'), df3['full_text'], df3['google_trans'])

df3['en_ja_goog'] = df3['en_ja_goog'].astype(str)
#### ML Preprocessing ######
import nltk

import string

import re

#bokeh 

from bokeh.io import show, output_notebook

from bokeh.models import ColumnDataSource, Panel, Tabs, FactorRange

from bokeh.models import HoverTool

from bokeh.plotting import figure

from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

from bokeh.palettes import Spectral5

import bokeh.layouts as layouts

from bokeh.layouts import row

import bokeh.models.widgets as widgets

from bokeh.io import curdoc

from bokeh.transform import factor_cmap

from bokeh.transform import dodge

output_notebook()
#extract link for tweet

def tweet_link(link):

    words = link.split()

    links = [word for word in words if word.startswith('http')]

    return links



#extract Hashtags

def hashtags(string):

    words = string.split()

    hashtags = [word for word in words if word.startswith('#')]

    return hashtags



#hashtag count

def hash_count(string):

    words = string.split()

    hashtags = [word for word in words if word.startswith('#')]

    return len(hashtags)



#Extract mention

def mention(string):

    words = string.split()

    mention = [word for word in words if word.startswith('@')]

    return mention



#extracting the emojis

emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())

r = re.compile('|'.join(re.escape(p) for p in emojis_list))
df3['tweet_link'] = df3['en_ja_goog'].apply(tweet_link)

df3['hashtag'] = df3['en_ja_goog'].apply(hashtags)

df3['hash_count'] = df3['en_ja_goog'].apply(hash_count)

df3['mention'] = df3['text'].apply(mention)

df3['emoji'] = df3['full_text'].str.findall(r)

df3['emoji_count'] = df3['emoji'].apply(len)
df3.head(10)
stopwords = nltk.corpus.stopwords.words('english')

wn = nltk.WordNetLemmatizer()

punc = lambda x: re.sub("!|,|\?|\'|-|\"|&|。|\)|\(|！|，|\.*|/|\[|\]|\u2026|\d|:|~|、|？|☆|’|– |【|】|「|」|《|》|※| “|”|＊|→||[\b\.\b]{3}||@||@ |#|# |", '',x)



#Clean

def clean_text(soup):

    soup = BeautifulSoup(soup, 'lxml')

    souped = soup.get_text()

    stripped = re.sub(r'https?://[A-Za-z0-9./]+', '', souped)

    words = stripped.split() 

    mention = [word for word in words if not word.startswith('@')]

    RT = [word for word in mention if not word.startswith('RT')]

    text = " ".join([wn.lemmatize(word) for word in RT if word not in stopwords])

    punct = "".join([word.lower() for word in text if word not in string.punctuation])

    short_words = ' '.join([w for w in punct.split() if len(w)>2])

    ja_punct = ''.join([punc(word) for word in short_words])

    tokens = re.split('\W+', ja_punct)

    return (" ".join(tokens)).strip()
#Cleaning the text 

df3['clean_text'] = df3['en_ja_goog'].apply(clean_text)
#To R to get sentiment and emotions 

# df3['clean_text'].to_csv('clean_cny_text.csv')
#Add in our R sentiment for range of emotion

r_sent_word = pd.read_csv('../input/distweet/r_emotion_cny.csv', index_col = 0)

r_sent_word = r_sent_word.reset_index(drop = True)

r_sent_word_counts = pd.DataFrame(r_sent_word.astype(bool).sum().reset_index())

r_sent_word_counts = r_sent_word_counts.rename(columns = {'index':'sentiment',0:'count'})
#Bokeh Chart 

sentiment = list(r_sent_word_counts['sentiment'])

count = list(r_sent_word_counts['count'])





source = ColumnDataSource(data=dict(sentiment=sentiment, count=count))



p = figure(x_range=sentiment, y_range=(0,1000),  plot_width=700, plot_height=700, title="R Sentiment Count",

           toolbar_location='below', tools="pan,wheel_zoom,box_zoom,reset")





p.vbar(x='sentiment', top='count', width=0.9, color = 'red', source=source)

p.title.align = 'center'

p.xgrid.grid_line_color = None

p.xaxis.major_label_orientation = "vertical"

p.left[0].formatter.use_scientific = False

p.add_tools(HoverTool(tooltips=[("sentiment", "@sentiment"), ("Total Count", "@count")]))





p.title.text_font_size = '20pt'

p.xaxis.axis_label="Sentiment"

p.xaxis.axis_label_text_font_size = "15pt"

p.xaxis.major_label_text_font_size = "15pt"

p.xaxis.axis_label_text_color = "black"

p.yaxis.axis_label="Count"

p.yaxis.axis_label_text_font_size = "15pt"

p.yaxis.major_label_text_font_size = "15pt"

p.yaxis.axis_label_text_color = "black"





tbsource = ColumnDataSource(r_sent_word_counts)



columns = [TableColumn(field = 'sentiment', title = 'Sentiment'),

          TableColumn(field = 'count', title = 'Count')]

data_table = DataTable(source = tbsource, columns = columns, width = 700, height = 700)



layout = row(p, data_table)



show(layout)
df3 = pd.concat([df3,r_sent_word],axis =1)
# load in our R tie breaker sentiment

r_sent = pd.read_csv('../input/distweet/r_sent_cny.csv', index_col=0)

r_sent = r_sent.reset_index(drop = True)

df3['r_sentiment'] = r_sent
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# analyser = SentimentIntensityAnalyzer()



# def sentiment_analyzer_scores(sentence):

#     score = analyser.polarity_scores(sentence)

#     if score['compound']>= 0.05:

#         return 'positive'

#     elif score['compound'] <= -0.05:

#         return 'negative'

#     else:

#         return 'neutral'

    

# from textblob import TextBlob



# def get_tweet_sentiment(tweet):

#     # create TextBlob object of passed tweet text

#     analysis = TextBlob(tweet)

#     # set sentiment

#     if analysis.sentiment.polarity > 0:

#         return 'positive'

#     elif analysis.sentiment.polarity == 0:

#         return 'neutral'

#     else:

#         return 'negative'



# vader = lambda x: sentiment_analyzer_scores(x)

# df3['vader_score_word'] = [vader(x) for x in df3['clean_text']]

# df3['tb_word'] = df3['clean_text'].apply(lambda tweet: get_tweet_sentiment(tweet))
# #Creating the tie breaker between textblob,vader and r_sentiment



# test = np.where((df3['vader_score_word'] == df3['tb_word']), df3['vader_score_word'], 'not sure')

# test2 = np.where((df3['tb_word'] == df3['r_sentiment']), df3['tb_word'], 'not sure')

# test3 = np.where((df3['vader_score_word'] == df3['r_sentiment']), df3['vader_score_word'], 'not sure')





# test = pd.DataFrame(test)

# test.rename(columns = {0:'vaderVstb'}, inplace = True)

# test['tbVsr'] = test2

# test['vaderVsr'] = test3





# t2 = np.where(test['vaderVstb'] == 'not sure', test['tbVsr'], test['vaderVstb'])

# t2 = pd.DataFrame(t2)

# t2.rename(columns = {0:'compare1'}, inplace = True)

# t2['compare2'] = np.where(t2['compare1'] == 'not sure', test['vaderVsr'], t2['compare1'])



# #Remaining 5510 rows still not sure.  Since vader analyzes emoticons better we will use vader as the final tie breaker

# t2['compare3'] = np.where(t2['compare2'] == 'not sure', df3['vader_score_word'], t2['compare2'])

# df3.loc[:,'label'] = t2.loc[:,'compare3']
# #Feature Creation 

# def count_punct(text):

#     count = sum([1 for char in text if char in string.punctuation])

#     return count

# #number of capitlizatized words

# def count_cap(text):

#     count = sum([1 for c in text if c.isupper()])

#     return count
# #Punctuation Count

# df3['punc_count'] = df3['full_text'].apply(lambda tweet: count_punct(tweet))

# #counting length of tweet

# df3['tweet_len'] = df3['en_ja_goog'].apply(lambda tweet: len(tweet) - tweet.count(' '))

# #Capitlization Count

# df3['cap_count'] = df3['en_ja_goog'].apply(count_cap)
# df4 = df3.loc[:,['full_text', 'clean_text', 'hash_count', 'emoji_count','anger','anticipation','disgust','fear','joy',

#           'sadness', 'surprise', 'trust','label','punc_count','tweet_len','cap_count','lang']]
# #lets set our categories as categories 

# df4['label'] = df4['label'].astype('category')

# df4['lang'] = df4['lang'].astype('category')

# #2 is positive, 1 is neutral, 0 is negative

# df4['label'] = df4['label'].cat.codes

# df_ja = df4[df4['lang'] == 'ja']

# df_en = df4[df4['lang'] == 'en']

# df_en['label'] = df_en['label'].replace(0,3)

# df_en['label'] = df_en['label'].replace(1,4)

# df_en['label'] = df_en['label'].replace(2,5)

# df5 = pd.concat([df_ja,df_en])
### ML Classification Model ###
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

import pickle

from collections import Counter
df5 = pd.read_csv("../input/ml-data/kaggle.csv")
df5['clean_text'] = df5['clean_text'].replace(np.nan,'')
my_stop_words = ENGLISH_STOP_WORDS.union(['disneyland','tokyo','disney', 'im', 'tdrnow','paris','california','amp','disneysea','got',

                                         'ºc', 'ºf', 'ºoº','𝗧𝗵𝗲','くまのプーさん', 'ディズニー', 'ディズニーシー','ディズニーハロウィーン',

                                         'ディズニーランド', 'ディズニー好きと繋がりたい', 'フェスティバルオブミスティーク', 'マルマン',

                                         'ㅋㅋㅋ', '場所', '更新', '月released', '東京ディズニーシー', '東京ディズニーランド', '東京ディズニーリゾート',

                                         '香港迪士尼樂園', 'ºº', 'hong', 'kong',"disneylandresort", "disneyland", "disneyresort",

                                          "californiaadventure",'downtowndisney','disneyanaheim','disneylandanaheim',

                                          'disneycalifornia','californiadisney','disneysea', 'disneytokyo', 'disneytokyoresort', 

                                          'tokyodisney','tokyodisneyresort', 'tokyodisneyland','東京ディズニーランド', 'ディズニーランド',

                                          '東京ディズニーシー', 'ズニーシー', 'tdr_now', 'tdr_md','tdr','dca','dl','tdrmd'])
ngram_vect = CountVectorizer(ngram_range=(1,3), max_features = 5000, stop_words = my_stop_words)

ngram = ngram_vect.fit_transform(df5['clean_text'])

X_ngram_df = pd.DataFrame(ngram.A, columns = ngram_vect.get_feature_names())

X_ngram_df = X_ngram_df.transpose()
#top 20 words for each retweet 

top_dict = {}

for c in X_ngram_df.columns:

    top = X_ngram_df[c].sort_values(ascending=False).head(20)

    top_dict[c]= list(zip(top.index, top.values))

#word count

n_words = []

for tweet in X_ngram_df.columns:

    top = [word for (word, count) in top_dict[tweet] if count != 0]

    for t in top:

        n_words.append(t)
count = pd.DataFrame(Counter(n_words).most_common())[0:20]

count.rename(columns={0:'word',1:'count'}, inplace = True)
#Bokeh Chart 

word = list(count['word'])

counts = list(count['count'])



source = ColumnDataSource(data=dict(word=word, counts=counts))



p = figure(x_range=word, y_range=(0,150),  plot_width=700, plot_height=500, title="Top 20 words",

           toolbar_location='below', tools="pan,wheel_zoom,box_zoom,reset")





p.vbar(x='word', top='counts', width=0.9, color = 'teal', source=source)

p.title.align = 'center'

p.xgrid.grid_line_color = None

p.xaxis.major_label_orientation = "vertical"

p.left[0].formatter.use_scientific = False

p.add_tools(HoverTool(tooltips=[("Word", "@word"), ("Total Count", "@counts")]))



p.title.text_font_size = '20pt'

p.xaxis.axis_label="Word"

p.xaxis.axis_label_text_font_size = "15pt"

p.xaxis.major_label_text_font_size = "15pt"

p.xaxis.axis_label_text_color = "black"

p.yaxis.axis_label="Count"

p.yaxis.axis_label_text_font_size = "15pt"

p.yaxis.major_label_text_font_size = "15pt"

p.yaxis.axis_label_text_color = "black"





tbsource = ColumnDataSource(count)



columns = [TableColumn(field = 'word', title = 'word'),

          TableColumn(field = 'count', title = 'Count')]

data_table = DataTable(source = tbsource, columns = columns, width = 700, height = 700)



layout = row(p, data_table)



show(layout)
###Pickling the Random Forest model ####
from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_fscore_support 

import time
y_test = df5['label']

count_vecto = CountVectorizer(stop_words = my_stop_words, max_features = 5000)

count_test = count_vecto.fit_transform(df5['clean_text'])

X_test_vect = pd.concat([df5[['hash_count', 'emoji_count', 'anger','anticipation','disgust','fear',

                                                       'joy', 'sadness', 'surprise', 'trust', 'punc_count',

                                                       'tweet_len','cap_count']].reset_index(drop=True), 

           pd.DataFrame(count_test.toarray())], axis=1)
X_test_vect.shape
# load the model from disk

filename = '../input/models/finalized_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model)
#Model Predict 

start = time.time()

y_pred = loaded_model.predict(X_test_vect)

end = time.time()

pred_time = (end - start)



#Model Scoring 

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='macro')

print('Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(

    round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))
importances = loaded_model.feature_importances_

(sorted(zip(importances, X_test_vect.columns), reverse=True))[0:20]
from sklearn.metrics import classification_report

target_names = ['ja_neg', 'ja_neu', 'ja_pos', 'en_neg', 'en_neu', 'en_pos']

print(classification_report(y_test, y_pred, target_names = target_names))
#Confusion matrix 

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

import seaborn as sns

cf_matrix = confusion_matrix(y_test, y_pred)

class_names = ['neg_ja', 'neu_ja', 'pos_ja', 'neg_en', 'neu_en', 'pos_en']





def plot_cm(y_true, y_pred, figsize=(15,10)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=(15,10))

    ax = sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

    bottom, top = ax.get_ylim()

    ax.set_ylim(bottom + 0.5, top - 0.5)

    

plot_cm(y_test, y_pred)

plt.show()
#ROC AUC

import scikitplot as scikitplot #to make things easy

y_pred_proba = loaded_model.predict_proba(X_test_vect)

scikitplot.metrics.plot_roc(y_test, y_pred_proba, figsize=(15,10))

plt.show()
from sklearn.metrics import roc_auc_score



y_prob = loaded_model.predict_proba(X_test_vect)



macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",

                                  average="macro")

weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",

                                     average="weighted")

macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",

                                  average="macro")

weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",

                                     average="weighted")

print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "

      "(weighted by prevalence)"

      .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))

print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "

      "(weighted by prevalence)"

      .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
###LDA Topic Modeling###
import sys

# !{sys.executable} -m spacy download en

import re, numpy as np, pandas as pd

from pprint import pprint



# Gensim

import gensim, spacy, logging, warnings

import gensim.corpora as corpora

from gensim.utils import lemmatize, simple_preprocess

from gensim.models import CoherenceModel





# NLTK Stop words

from nltk.corpus import stopwords

stop_words = stopwords.words('english')

stop_words.extend(['disneyland','tokyo','disney', 'im', 'tdrnow','paris','california','amp','disneysea','got',

                 'ºc', 'ºf', 'ºoº','𝗧𝗵𝗲','くまのプーさん', 'ディズニー', 'ディズニーシー','ディズニーハロウィーン',

                 'ディズニーランド', 'ディズニー好きと繋がりたい', 'フェスティバルオブミスティーク', 'マルマン',

                 'ㅋㅋㅋ', '場所', '更新', '月released', '東京ディズニーシー', '東京ディズニーランド', '東京ディズニーリゾート',

                 '香港迪士尼樂園', 'ºº', 'hong', 'kong',"disneylandresort", "disneyland", "disneyresort",

                  "californiaadventure",'downtowndisney','disneyanaheim','disneylandanaheim',

                  'disneycalifornia','californiadisney','disneysea', 'disneytokyo', 'disneytokyoresort', 

                  'tokyodisney','tokyodisneyresort', 'tokyodisneyland','東京ディズニーランド', 'ディズニーランド',

                  '東京ディズニーシー', 'ズニーシー', 'tdr_now', 'tdr_md','tdr','dca','dl', 'wdw','disneylandparis',

                  'theme_park', 'min', 'day', 'new', 'guy', 'year', 'way', 'part', 'thing', 'man','ティスニー',

                  'side', 'sia', 'ティスニーラント', 'ティスニーハロウィーン', 'today', 'wanna', 'place',

                  'world', 'disneyworld', 'next', 'disneypark', '東京ティスニーシー', 'yen',

                  '東京ティスニーラント', 'land', 'park', 'ティスニーシー', 'tdl', 'tdrmd', 'lot',

                  "東京ティスニーリソート"])



%matplotlib inline

warnings.filterwarnings("ignore",category=DeprecationWarning)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
def sent_to_words(sentences):

    for sent in sentences:

        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails

        sent = re.sub('\s+', ' ', sent)  # remove newline chars

        sent = re.sub("\'", "", sent)  # remove single quotes

        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 

        yield(sent)  



# !python3 -m spacy download en  # run in terminal once

def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""

    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    texts = [bigram_mod[doc] for doc in texts]

    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

    texts_out = []

    nlp = spacy.load('en', disable=['parser', 'ner'])

    for sent in texts:

        doc = nlp(" ".join(sent)) 

        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    # remove stopwords once more after lemmatization

    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    

    return texts_out
#For simpliclity lets not seperate positive and negative and by language

# Convert to list

data = df5.clean_text.values.tolist()

data_words = list(sent_to_words(data))       





# Build the bigram and trigram models

bigram = gensim.models.Phrases(data_words, min_count=2, threshold=20) # higher threshold fewer phrases. Must show up more than 2 times the phrases 

trigram = gensim.models.Phrases(bigram[data_words], threshold=20)  

bigram_mod = gensim.models.phrases.Phraser(bigram)

trigram_mod = gensim.models.phrases.Phraser(trigram)
data_ready = process_words(data_words)  # processed Text Data!

data_ready[0:10]
# Create Dictionary

id2word = corpora.Dictionary(data_ready)



# Create Corpus: Term Document Frequency

corpus = [id2word.doc2bow(text) for text in data_ready]
#4 Topics

lda_model4 = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=4, 

                                           random_state=777,

                                           update_every=1,

                                           chunksize=10,

                                           passes=10,

                                           alpha='symmetric',

                                           iterations=20,

                                           per_word_topics=True)
pprint(lda_model4.print_topics())
#7 Topics

lda_model7 = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=7, 

                                           random_state=777,

                                           update_every=1,

                                           chunksize=10,

                                           passes=10,

                                           alpha='symmetric',

                                           iterations=20,

                                           per_word_topics=True)
pprint(lda_model7.print_topics())
#Final Model 



#4 Topics

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,

                                           id2word=id2word,

                                           num_topics=4, 

                                           random_state=777,

                                           update_every=1,

                                           chunksize=10,

                                           passes=10,

                                           alpha='symmetric',

                                           iterations=100,

                                            per_word_topics=True)
# 1. Wordcloud of Top N words in each topic

from matplotlib import pyplot as plt

from wordcloud import WordCloud, STOPWORDS

import matplotlib.colors as mcolors



cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



cloud = WordCloud(stopwords=stop_words,

                  background_color='black',

                  width=2500,

                  height=1800,

                  max_words=20,

                  colormap='tab10',

                  color_func=lambda *args, **kwargs: cols[i],

                  prefer_horizontal=1.0)



topics = lda_model.show_topics(formatted=False)



fig, axes = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)



for i, ax in enumerate(axes.flatten()):

    fig.add_subplot(ax)

    topic_words = dict(topics[i][1])

    cloud.generate_from_frequencies(topic_words, max_font_size=300)

    plt.gca().imshow(cloud)

    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))

    plt.gca().axis('off')





plt.subplots_adjust(wspace=0, hspace=0)

plt.axis('off')

plt.margins(x=0, y=0)

plt.tight_layout()

plt.show()
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):

    # Init output

    sent_topics_df = pd.DataFrame()



    # Get main topic in each document

    for i, row_list in enumerate(ldamodel[corpus]):

        row = row_list[0] if ldamodel.per_word_topics else row_list            

        # print(row)

        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        # Get the Dominant topic, Perc Contribution and Keywords for each document

        for j, (topic_num, prop_topic) in enumerate(row):

            if j == 0:  # => dominant topic

                wp = ldamodel.show_topic(topic_num)

                topic_keywords = ", ".join([word for word, prop in wp])

                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)

            else:

                break

    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']



    # Add original text to the end of the output

    contents = pd.Series(texts)

    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return(sent_topics_df)





df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)



# Format

df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
import seaborn as sns

import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'



fig, axes = plt.subplots(2,2,figsize=(10,8), dpi=160, sharex=True, sharey=True)



for i, ax in enumerate(axes.flatten()):    

    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]

    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]

    ax.hist(doc_lens, bins = 1000, color=cols[i])

    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])

    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())

    ax.set(xlim=(0, 30), xlabel='Document Word Count')

    ax.set_ylabel('Number of Documents', color=cols[i])

    ax.set_title('Topic: '+str(i), fontdict=dict(size=12, color=cols[i]))



fig.tight_layout()

fig.subplots_adjust(top=0.90)

fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=14)

plt.show()
# Get topic weights and dominant topics ------------

from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook



# Get topic weights

topic_weights = []

for i, row_list in enumerate(lda_model[corpus]):

    topic_weights.append([w for i, w in row_list[0]])



# Array of topic weights    

arr = pd.DataFrame(topic_weights).fillna(0).values



# Keep the well separated points (optional)

arr = arr[np.amax(arr, axis=1) > 0.35]



# Dominant topic number in each doc

topic_num = np.argmax(arr, axis=1)



# tSNE Dimension Reduction

tsne_model = TSNE(n_components=2, verbose=1, random_state= 77, angle=.99, init='pca')

tsne_lda = tsne_model.fit_transform(arr)
def explore_topic(lda_model, topic_number, topn, output=True):

    """

    accept a ldamodel, atopic number and topn vocabs of interest

    prints a formatted list of the topn terms

    """

    terms = []

    for term, frequency in lda_model.show_topic(topic_number, topn=topn):

        terms += [term]

        if output:

            print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))

    

    return terms
num_topics = 4

topic_summaries = []

print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')

for i in range(num_topics):

    print('Topic '+str(i)+' |---------------------\n')

    tmp = explore_topic(lda_model,topic_number=i, topn=10, output=True )

#     print tmp[:5]

    topic_summaries += [tmp[:5]]

    print
# Plot the Topic Clusters using Bokeh

output_notebook()

n_topics = 4

mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 

              plot_width=900, plot_height=700,tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave")

plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])



show(plot)
import pyLDAvis.gensim

pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)

vis
import pandas as pd

import json

from wordcloud import WordCloud,STOPWORDS 

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
list_tweets = []

with open('../input/twitter-data/twitter_data.txt','r') as td:

    for lin in td:

        try:

            lin = lin.strip()

            line = json.loads(lin)

            if type(line) is dict:

                list_tweets.append(line)

        except:

            pass
total_tweets = len(list_tweets)
df = pd.DataFrame(list_tweets)
df.drop(['contributors','coordinates'],axis=1,inplace=True)
df.info()
df.lang.unique()
count = pd.DataFrame(pd.value_counts(df['lang'])).reset_index()

count
count.plot(kind='bar',x='index')
df_final = df[df['lang'] == 'en']
df_final
df_final['created_at'] = pd.to_datetime(df_final.created_at)
print(df_final.display_text_range.isna().sum())

len(df_final.display_text_range)

df_final.drop('display_text_range',inplace = True,axis=1)
print(df_final.extended_entities.isna().sum())

len(df_final.extended_entities)

df_final.drop('extended_entities',inplace = True,axis=1)
print(df_final.favorite_count.sum())

len(df_final.favorite_count)

df_final.drop('favorite_count',inplace = True,axis=1)
print(df_final.favorited.sum())

len(df_final.favorited)

df_final.drop('favorited',inplace = True,axis=1)
df_final.drop(['filter_level','geo','id','id_str','in_reply_to_screen_name','in_reply_to_status_id','in_reply_to_status_id_str'],inplace = True,axis=1)
df_final.drop(['in_reply_to_user_id','in_reply_to_user_id_str','is_quote_status'],inplace=True,axis=1)
df_final.place.isna().sum()

len(df_final.place)

df_final.drop(['place','possibly_sensitive','quote_count','quoted_status','quoted_status_id','quoted_status_id_str','quoted_status_permalink'],inplace = True,axis=1)
df_final.drop(['reply_count','retweet_count','retweeted'],inplace=True,axis=1)

df_final
df_final.drop(['truncated','timestamp_ms'],inplace=True,axis=1)

df_final.drop('entities',inplace=True,axis=1)
df_final.drop('user',inplace=True,axis=1)
df_final.extended_tweet.iloc[6]
tweets = pd.DataFrame(list(df_final.extended_tweet[df_final.extended_tweet.isna() == False]))
tweets = list(tweets.full_text)
tweets
for t in range(10):

    print(tweets[t],end='\n\n\n\n')
stopwords = set(STOPWORDS)

stopwords.add('https')

stopwords.add('amp')

stopwords.add('will')

words = []

for i in tweets:

    for j in i.split(' '):

            words.append(j)

sentence = ''

for i in words:

        sentence = sentence + i + ' '
wordcloud = WordCloud(width = 1500, height = 800,

                      background_color ='white',

                      stopwords = stopwords,

                      min_font_size=10).generate(sentence)
plt.figure(figsize = (100,80), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
wordcloud.to_file("first_review.png")
def strip_punctuation(word):

    for i in punctuation_chars:

        word = word.replace(i,'')

    return word



def get_neg(s):

#     s = s.split(' ')

    n_list = []

    c=0

    for i in s:

        if strip_punctuation(i) in negative_words:

            c += 1

            n_list.append(strip_punctuation(i))

    return c,n_list



def get_pos(s):

#     s = s.split(' ')

    p_list = []

    c=0

    for i in s:

        if strip_punctuation(i) in positive_words:

            c += 1

            p_list.append(strip_punctuation(i))

    return c,p_list
# lists of words to use

positive_words = []

with open("../input/twitter-data/positive-words.txt") as pos_f:

    for lin in pos_f:

        if lin[0] != ';' and lin[0] != '\n':

            positive_words.append(lin.strip())



negative_words = []

with open("../input/twitter-data/negative-words.txt") as pos_f:

    for lin in pos_f:

        if lin[0] != ';' and lin[0] != '\n':

            negative_words.append(lin.strip())
punctuation_chars = ["'", '"', ",", ".", "!", ":", ";", '#', '@']

p_score,p_list = get_pos(words)

n_score,n_list = get_neg(words)

net_score = p_score - n_score

f_str = 'Postive score = {}, Negative score = {}, Net score = {}'.format(p_score,n_score,net_score)
f_str
sentence = ''

for i in p_list:

        sentence = sentence + i + ' '

        

wordcloud = WordCloud(width = 1500, height = 800,

                      background_color ='white',

                      stopwords = stopwords,

                      min_font_size=10).generate(sentence)



plt.figure(figsize = (100,80), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 

wordcloud.to_file("postive_tweetWords.png")
sentence = ''

for i in n_list:

        sentence = sentence + i + ' '

        

wordcloud = WordCloud(width = 1500, height = 800,

                      background_color ='white',

                      stopwords = stopwords,

                      min_font_size=10).generate(sentence)



plt.figure(figsize = (100,80), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 

wordcloud.to_file("negative_tweetWords.png")
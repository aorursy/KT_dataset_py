import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
from wordcloud import WordCloud
from sklearn.decomposition import NMF, LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
tweets = pd.read_csv('/kaggle/input/customer-support-on-twitter/twcs/twcs.csv')
tweets[0:10]
support_services = tweets['author_id'].value_counts()
support_services.sort_index(ascending=False,inplace=True)
support_services = support_services[:108]
%matplotlib inline

plt.figure(figsize=(30,20))
plt.title('Support interactions by brands')
plt.xlabel('Brands')
plt.ylabel('Interactions')
plt.bar(list(support_services.index.values),list(support_services.values))
plt.xticks(rotation=90)
plt.show()
chase_support = tweets[tweets['author_id']=='ChaseSupport']
chase_support.head()
chase_support['interaction_tweet'] = ''
chase_support['interaction_author'] = ''
chase_support['interaction_tweet_at'] = ''
chase_support.dropna(subset=['tweet_id', 'in_response_to_tweet_id'],inplace=True)
chase_support.reset_index(drop=True,inplace=True)
for i in range(len(chase_support)):
    chase_support.at[i, 'interaction_tweet'] = str(list(tweets[tweets['tweet_id']==int(chase_support.iloc[i]['in_response_to_tweet_id'])]['text'])).strip("[']")
    chase_support.at[i, 'interaction_author'] = str(list(tweets[tweets['tweet_id']==int(chase_support.iloc[i]['in_response_to_tweet_id'])]['author_id'])).strip("[']") 
    chase_support.at[i, 'interaction_tweet_at'] = str(list(tweets[tweets['tweet_id']==int(chase_support.iloc[i]['in_response_to_tweet_id'])]['created_at'])).strip("[']")
chase_support.head()
chase_support.iloc[0]['interaction_tweet']
chase_support.iloc[1]['text']
chase_support.iloc[0]['text']
chase_support.dropna(subset=['interaction_tweet_at'],inplace=True)
chase_support.shape
days = chase_support['interaction_tweet_at'].str.extractall(r'(^\w+)').values
%matplotlib inline
plt.figure(figsize=(20,10))
plt.title('Interactions by day')
plt.xlabel('Days of the week')
plt.ylabel('Interactions')
plt.hist(days,bins=7,histtype='stepfilled',density=True)
plt.show()
time = chase_support['interaction_tweet_at'].str.extractall(r'(\s\d{1,2}[:])').values
hours = []
for i in range(len(time)):
    hours.append(int(time[i][0].strip(' :')))
%matplotlib inline
plt.figure(figsize=(20,10))
plt.title('Interactions by day')
plt.xlabel('Hours of the day')
plt.ylabel('Interactions')
plt.hist(hours,histtype='stepfilled',density=True)
plt.show()
dates = chase_support['interaction_tweet_at'].str.extractall(r'(\D{3}\s\d{1,2})').values
days_of_month = []
for i in range(len(dates)):
    days_of_month.append(int(dates[i][0][-2:]))
days_of_month.sort()
%matplotlib inline
plt.figure(figsize=(20,10))
plt.title('Interactions by days of the month')
plt.xlabel('Days of the month')
plt.ylabel('Interactions')
plt.hist(days_of_month,histtype='stepfilled',density=True)
plt.show()
chase_support['Client interaction time'] = chase_support['interaction_tweet_at'].str.extract(r'(\d{1,2}[:]\d{1,2}[:]\d{1,2})').values
chase_support['Support interaction time'] = chase_support['created_at'].str.extract(r'(\d{1,2}[:]\d{1,2}[:]\d{1,2})').values
chase_support['TTR'] = np.nan
chase_support.drop(chase_support.index[[8348,436]],inplace=True)
chase_support.reset_index(drop=True,inplace=True)
for i in range(len(chase_support)):
    chase_support.at[i, 'TTR'] = pd.Timestamp(chase_support.iloc[i]['created_at']) - pd.Timestamp(chase_support.iloc[i]['interaction_tweet_at'])
chase_support.dropna(subset=['TTR'],inplace=True)
chase_support.reset_index(drop=True,inplace=True)
str(chase_support['TTR'].astype('timedelta64[s]').mean())
str(chase_support['TTR'].astype('timedelta64[s]').max())
chase_support.sort_values(by=['TTR'],ascending=False)[0:10]
print(chase_support.iloc[7811]['interaction_tweet'])
print(chase_support.iloc[8699]['interaction_tweet'])
print(chase_support.iloc[7814]['interaction_tweet'])
print(chase_support.iloc[7809]['interaction_tweet'])
print(chase_support.iloc[7840]['interaction_tweet'])
print(chase_support.iloc[7760]['interaction_tweet'])
print(chase_support.iloc[7753]['interaction_tweet'])
print(chase_support.iloc[7645]['interaction_tweet'])
print(chase_support.iloc[7649]['interaction_tweet'])
chase_support["interaction_tweet"] = chase_support["interaction_tweet"].str.lower()
chase_support.head()
", ".join(stopwords.words('english'))
for i in range(len(chase_support)):
    chase_support.at[i,'interaction_tweet'] = chase_support.loc[i]['interaction_tweet'].replace('"','')
stops = set(stopwords.words('english'))
def remove_stops(corpus):
    return " ".join([word for word in str(corpus).split() if word not in stops])
chase_support["interaction_tweet"] = chase_support["interaction_tweet"].apply(lambda text: remove_stops(text))
chase_support.head()
punctuation_signs = string.punctuation
def remove_punctuation_signs(corpus):
    return corpus.translate(str.maketrans('', '', punctuation_signs))

chase_support["interaction_tweet"] = chase_support["interaction_tweet"].apply(lambda corpus: remove_punctuation_signs(corpus))
chase_support.head()
for i in range(len(chase_support)):
    chase_support.at[i,'interaction_tweet'] = chase_support.loc[i]['interaction_tweet'].replace('chasesupport','')
chase_support.head()
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_tweets(corpus):
    pos_tagged_text = nltk.pos_tag(corpus.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

chase_support['interaction_tweet'] = chase_support['interaction_tweet'].apply(lambda corpus: lemmatize_tweets(corpus))
chase_support.head()
counter = Counter()
for text in chase_support["interaction_tweet"].values:
    for word in text.split():
        counter[word] += 1
        
counter.most_common(15)
del counter['card']
del counter['bank']
del counter['account']
del counter['credit']
del counter['money']
del counter['service']
del counter['app']
del counter['fraud']
del counter['atm']
del counter['debit']
del counter['branch']
del counter['rewards']
del counter['sapphire']
del counter['website']
del counter['good']
del counter['bad']
del counter['fuck']
del counter['worst']
del counter['email']
del counter['banking']
del counter['deposit']
del counter['support']
del counter['cash']
del counter['fees']
del counter['fucking']
del counter['issue']
del counter['fucking']
del counter['love']
del counter['transaction']
del counter['ultimate']
del counter['transfer']
del counter['password']
del counter['mobile']
del counter['problem']
del counter['phone']
del counter['purchase']
del counter['fix']
del counter['lose']
del counter['business']
del counter['number']
del counter['never']
del counter['pay']
del counter['check']
del counter['reward']
del counter['cancel']
del counter['error']
del counter['fee']
del counter['payment']
del counter['bill']
del counter['balance']
del counter['shit']
del counter['suck']
del counter['mail']
del counter['customer']
del counter['charge']
del counter['call']
del counter['help']
del counter['time']
del counter['new']
del counter['work']
del counter['close']
del counter['open']
del counter['online']
del counter['wait']
counter.most_common(50)
frequent_words = set([w for (w, wc) in counter.most_common(50)])
def remove_frequent_words(corpus):
    return " ".join([word for word in str(corpus).split() if word not in frequent_words])

chase_support['interaction_tweet'] = chase_support['interaction_tweet'].apply(lambda corpus: remove_frequent_words(corpus))
chase_support.head()
comment_words = '' 

for val in chase_support['interaction_tweet']:     
    val = str(val) 
    tokens = val.split() 
    comment_words += " ".join(tokens)+" "
  
wordcloud = WordCloud(width = 1300,height = 800,background_color ='white',min_font_size = 10).generate(comment_words) 
                    
plt.figure(figsize = (20, 20), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
from sklearn.feature_extraction.text import TfidfVectorizer

#We'll implement the ngram_range the way it is because this way the corpus becomes more informative. It's not the same to say 'fraud' than 'Chase is fraud', right?
vect = TfidfVectorizer(ngram_range=(2,3), max_features=20000).fit(chase_support['interaction_tweet'])
data = vect.fit_transform(chase_support['interaction_tweet'])
vect_feature_names = vect.get_feature_names()
# NMF : Non-negative matrix factorization (NMF or NNMF),
nmf = NMF(n_components=8, random_state=0, alpha=.1, l1_ratio=.5,max_iter=10000).fit(data)
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 20
print('NMF')
display_topics(nmf, vect_feature_names, no_top_words)
# Create Document — Topic Matrix
nmf_output = nmf.transform(data)
# column names
topicnames = ['Topic' + str(i) for i in range(nmf.n_components)]
# index names
docnames = ['Doc' + str(i) for i in range(len(chase_support))]
# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(nmf_output, 2), columns=topicnames, index=docnames)
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

df_document_topics = df_document_topic
df_document_topics.reset_index(inplace=True,drop=True)
chase_support['label'] = df_document_topics['dominant_topic']
chase_support = chase_support[['interaction_tweet','label']]
chase_support.drop_duplicates(subset ="interaction_tweet",keep = False, inplace = True) 
len(chase_support)

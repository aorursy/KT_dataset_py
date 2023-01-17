%%capture
!pip install --upgrade -q comet_ml contractions emoji unidecode langdetect
# # import comet_ml in the top of your file
# from comet_ml import Experiment
%%capture
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import emoji
import plotly.graph_objects as go
import nltk
import os
import string
import contractions
import xgboost
import time
import gc
import warnings

from langdetect import detect
from wordcloud import WordCloud
from urllib import request
from nltk.corpus import stopwords
from collections import Counter
from nltk import bigrams
from nltk import bigrams
from wordcloud import STOPWORDS
from nltk import word_tokenize, sent_tokenize, FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import RidgeClassifier

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
#warnings.filterwarnings('ignore')
#Import of Data for use in a Notebook
# train_df = pd.read_csv("https://raw.githubusercontent.com/Maddy-Muir/Classification_Predict/master/train.csv")
# test_df = pd.read_csv("https://raw.githubusercontent.com/Maddy-Muir/Classification_Predict/master/test.csv")

#Import of Data for use in Kaggle
train_df =pd.read_csv('../input/climate-change-belief-analysis/train.csv')
test_df =pd.read_csv('../input/climate-change-belief-analysis/test.csv')
train_df.info()
np.where(train_df.applymap(lambda x: x == ''))
train_df.head()
# Determining number of rows for each sentiment
rows = train_df['sentiment'].value_counts()
rows_df = pd.DataFrame({'Sentiment':rows.index, 'Rows':rows.values})

# Determining percentage distribution for each sentiment
percentage = round(train_df['sentiment'].value_counts(normalize=True)*100,2)
percentage_df = pd.DataFrame({'Sentiment':percentage.index,
                              'Percentage':percentage.values})

# Joining row and percentage information
sentiment_df = pd.merge(rows_df, percentage_df, on='Sentiment', how='outer')
sentiment_df.set_index('Sentiment', inplace=True)
sentiment_df.sort_index(axis = 0)
sns.countplot(x = 'sentiment', data = train_df, palette="hls")
plt.title("Sentiment Distribution");
# Separate joined words based on capitals
def camel_case_split(identifier):
    matches = re.finditer(
        r'.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
        identifier
    )
    return " ".join([m.group(0) for m in matches])

# Extract Mentions
def extract_mentions(tweet):
  """Helper function to extract mentions"""
  mentions = re.findall(r'@([a-zA-Z0-9_]{1}[a-zA-Z0-9_]{0,14})', tweet)
  return mentions

# Extract Hashtags
def extract_hash_tags(tweet):
  """Helper function to extract hashtags"""
  hash_tags = re.findall(r'(?:^|\s)#(\w+)', tweet)
  return [camel_case_split(tag) for tag in hash_tags]

# Identifying Retweets
def is_retweet(tweet):
  """Helper Function to determine if a tweet is a re-tweet"""
  match = re.search(r'^\s?RT\s@[a-zA-Z0-9_]{1}[a-zA-Z0-9_]{0,14}[\s:]', tweet)
  if match:
    return 1
  return 0

# Extract Emojis
def extract_emojis(tweet):
  """Helper to extract all emoji's from a tweet"""
  emojis = ''.join(c for c in tweet if c in emoji.UNICODE_EMOJI)
  if emojis:
    return emojis
  return None

# Language Detection of tweet
def detect_language(tweet):
  """Helpler function to detect language"""
  return detect(tweet)
# Extracting Mentions
train_df['mentions'] = train_df['message'].apply(extract_mentions)

# Extracting Hashtags
train_df['hashtags'] = train_df['message'].apply(extract_hash_tags)

# Identifying Retweets
train_df['is_retweet'] = train_df['message'].apply(is_retweet)

# Language Detection
train_df['language'] = train_df['message'].apply(detect_language)

# Extracting Emojis
train_df['emojis'] = train_df['message'].apply(extract_emojis)
train_df.emojis.fillna(value=np.nan, inplace=True)

# Finding Number of Words per Tweet
train_df["num_words"] = train_df["message"].apply(lambda x: len(str(x).split()))

# Finding Number of Characters per Tweet
train_df["num_chars"] = train_df["message"].apply(lambda x: len(str(x)))
train_df.head()
# Create a list of all the mentions
mentions_list = [item for sublist in train_df['mentions'] for item in sublist]

# Grouping mentions by sentiment
# News Mentions
news_mentions = train_df[train_df['sentiment'] == 2]['mentions']
news_mentions = [x for x in news_mentions if x != []]
news_mentions = [item for sublist in news_mentions for item in sublist]

# Positive Mentions
pos_mentions = train_df[train_df['sentiment'] == 1]['mentions']
pos_mentions = [x for x in pos_mentions if x != []]
pos_mentions = [item for sublist in pos_mentions for item in sublist]

# Neutral Mentions
neutral_mentions =train_df[train_df['sentiment'] == 0]['mentions']
neutral_mentions = [x for x in neutral_mentions if x != []]
neutral_mentions = [item for sublist in neutral_mentions for item in sublist]

# Negative Mentions
neg_mentions = train_df[train_df['sentiment'] ==-1]['mentions']
neg_mentions = [x for x in neg_mentions if x != []]
neg_mentions = [item for sublist in neg_mentions for item in sublist]
# Get count of mentions and unique mentions
print("Total number of mentions: \t\t\t"+ str(len(mentions_list)))
print("Total number of unique mentions: \t\t"+ str(len(set(mentions_list))))

# Get count of mentions and unique mentions per sentiment
print("Total number of News mentions: \t\t\t"+ str(len(news_mentions)))
print("Total number of unique News mentions: \t\t"+ str(len(set(news_mentions))))

print("Total number of Positve mentions: \t\t"+ str(len(pos_mentions)))
print("Total number of unique Positive mentions: \t"+ str(len(set(pos_mentions))))

print("Total number of Neutral mentions: \t\t"+ str(len(neutral_mentions)))
print("Total number of unique Neutral mentions: \t"+ str(len(set(neutral_mentions))))

print("Total number of Negative mentions: \t\t"+ str(len(neg_mentions)))
print("Total number of unique Negative mentions: \t"+ str(len(set(neg_mentions))))

# Count of common mentions
common_mentions = set(pos_mentions) & set(news_mentions) & set(neg_mentions) & set(neutral_mentions)
print("Total number of Common mentions: \t\t"+ str(len(common_mentions)))
mentions =['All', 'Postive', 'Neutral', 'Negative', 'News']

fig = go.Figure(data=[
    go.Bar(name='Total Mentions', x=mentions, y=[14799, 8497, 2198, 1386, 2718],marker_color='lightblue'),
    go.Bar(name='Unique Mentions', x=mentions, y=[7640, 4495, 1880, 919, 1302], marker_color ='purple')
])
# Change the bar mode
fig.update_layout(barmode='group', title = "Distribution of Mentions")

fig.show()
# Extract rows based on sentiment
HT_neg = train_df[train_df['sentiment'] == -1]['hashtags']
HT_neutral = train_df[train_df['sentiment'] == 0]['hashtags']
HT_pos = train_df[train_df['sentiment'] == 1]['hashtags']
HT_news = train_df[train_df['sentiment'] == 2]['hashtags']

# # List of sentiment hashtags
HT_neg = sum(HT_neg,[])
HT_neutral = sum(HT_neutral,[])
HT_pos = sum(HT_pos, [])
HT_news = sum(HT_news,[])
# Graph for Negative Sentiment
a = nltk.FreqDist(HT_neg)
d = pd.DataFrame({'Negative': list(a.keys()),
                  'Count': list(a.values())})
    
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Negative", y = "Count",palette="hls")
ax.set(ylabel = 'Count')
plt.title("Top 10 Negative Hashtags")
plt.show()

# Graph for Neutral Sentiment
a = nltk.FreqDist(HT_neutral)
d = pd.DataFrame({'Neutral': list(a.keys()),
                  'Count': list(a.values())})

d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Neutral", y = "Count",palette="hls")
ax.set(ylabel = 'Count')
plt.title("Top 10 Neutral Hashtags")
plt.show()

# Graph for Positive Sentiment
a = nltk.FreqDist(HT_pos)
d = pd.DataFrame({'Positive': list(a.keys()),
                  'Count': list(a.values())})
    
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Positive", y = "Count",palette="hls")
ax.set(ylabel = 'Count')
plt.title("Top 10 Positive Hashtags")
plt.show()

# Graph for News Sentiment
a = nltk.FreqDist(HT_news)
d = pd.DataFrame({'News': list(a.keys()),
                  'Count': list(a.values())})
     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "News", y = "Count",palette="hls")
ax.set(ylabel = 'Count')
plt.title("Top 10 News Hashtags")
plt.show()
# Determining number of rows which are retweets
rows_RT = train_df['is_retweet'].value_counts()
rows_RT_df = pd.DataFrame({'Is_Retweet':rows_RT.index, 'Rows':rows_RT.values})

# Determining percentage of rows which are retweets
percentage_RT = round(train_df['is_retweet'].value_counts(normalize=True)*100,2)
percentage_RT_df = pd.DataFrame({'Is_Retweet':percentage_RT.index, 
                                 'Percentage':percentage_RT.values})

# Joining row and percentage information
RT_df = pd.merge(rows_RT_df, percentage_RT_df, on='Is_Retweet', how='outer')
RT_df.set_index('Is_Retweet', inplace=True)
RT_df.sort_index(axis = 0)
# Extracting dataframe with duplicate messages
duplicates_df = train_df[train_df['message'].duplicated()]

# Checking how many duplicatas are not retweets
x = len(duplicates_df[duplicates_df['is_retweet'] == 0])
print("Total number of duplicate tweets which are NOT retweets: \t"+ str(x))
# Creating a dataframe with the count for each language
languages = train_df['language'].value_counts()
language_df = pd.DataFrame({'ISO Code':languages.index, 'Rows':languages.values})
language_df.set_index('ISO Code', inplace=True)
language_df.head()
# Extracting the rows that contains emojis
emoji_df = train_df[train_df['emojis'].notnull()]
print("Total number of tweets containing emojis: "+ str(len(emoji_df)))
print("Total number of emojis in tweets: \t  "+ str(len(''.join(emoji_df['emojis'].tolist()))))
def emoji_cloud(df):
  """ Create an emoji cloud
  Taken from: https://github.com/amueller/word_cloud/blob/d1ec087a7f86e6dc14ed3771a9f8e84a5d384e0a/examples/emoji.py
  """

  # Create a string with all the emojis
  emoji_list = ''.join(df['emojis'].tolist())

  # Get data directory (using getcwd() is needed to support running example in generated IPython notebook)
  d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

  # font_path = os.path.join(d, 'fonts', 'EmojiOneColor', 'EmojiOneColor.ttf')
  font_path = '../input/emoji-fonts/EmojiOneColor.otf'

  # Checking if font has been downloaded, if not download font
  if not os.path.isfile(font_path):
    if not os.path.exists(os.path.join(d, 'fonts', 'EmojiOneColor')):
      os.makedirs(os.path.join(d, 'fonts', 'EmojiOneColor'))
      
    url = 'https://github.com/adobe-fonts/emojione-color/blob/master/EmojiOneColor.otf?raw=true'
    request.urlretrieve(url, font_path)

  # Two consecutive punctuations :)
  ascii_art = r"(?:[{punctuation}][{punctuation}]+)".format(punctuation=string.punctuation)

  # A single character that is not alpha_numeric or other ascii printable
  emoji = r"(?:[^\s])(?<![\w{ascii_printable}])".format(ascii_printable=string.printable)

  # Regular expression matching ascii_art or emoji character
  regexp = r"{ascii_art}|{emoji}".format(ascii_art=ascii_art, emoji=emoji)

  # Create the Emoji-Cloud
  emojis = WordCloud(
    background_color='black',
    max_words=150,
    max_font_size=70, 
    scale=20,
    random_state=42,
    collocations=False,
    font_path=font_path,
    regexp=regexp,
    normalize_plurals=False
  ).generate(emoji_list)

  # Plot the Emoji-Cloud
  plt.figure(figsize=(12,8))
  plt.tight_layout(pad = 0) 
  plt.imshow(emojis)
  plt.axis('off')
  plt.title("Emoji Cloud", fontsize = 20)
  plt.show()
  plt.savefig('emoji_cloud.png')

emoji_cloud(emoji_df)
# Boxplot for Number of words in each class
f, axes = plt.subplots(2, 1, figsize=(9,12))
sns.boxplot(x='sentiment', y="num_chars", data=train_df, ax=axes[0], palette="hls")
axes[0].set_xlabel('sentiment', fontsize=12)
axes[0].set_title("Number of Words in each Class", fontsize=15)

# Boxplot for Number of characters in each class
sns.boxplot(x='sentiment', y='num_words', data=train_df, ax=axes[1], palette="hls")
axes[1].set_xlabel('sentiment', fontsize=12)
axes[1].set_title("Number of Characters in each Class", fontsize=15);
# Removing words that has no relevance to the context (https, RT, CO)
train_df['word_cloud'] = train_df['message'].str.replace('http\S+|www.\S+', '', case=False)

# Removing common words which appear in all sentiments
remove_words = ['climate', 'change', 'RT', 'global', 'warming', 'Donald', 'Trump']

# Function to remove common words listed above
def remove_common_words(message):
  pattern = re.compile(r'\b(' + r'|'.join(remove_words) + r')\b\s*')
  message = pattern.sub('', message)
  return message

train_df['word_cloud'] = train_df['word_cloud'].apply(remove_common_words)

# Extracing rows per sentiment
news = train_df[train_df['sentiment'] == 2]['word_cloud']
pos = train_df[train_df['sentiment'] == 1]['word_cloud']
neutral = train_df[train_df['sentiment'] == 0]['word_cloud']
neg = train_df[train_df['sentiment'] ==-1]['word_cloud']

# Splitting strings into lists
news = [word for line in news for word in line.split()]
pos = [word for line in pos for word in line.split()]
neutral = [word for line in neutral for word in line.split()]
neg = [word for line in neg for word in line.split()]

news = WordCloud(
    background_color='black',
    max_words=100,
    max_font_size=60, 
    scale=20,
    random_state=42,
    collocations=False,
    normalize_plurals=False
).generate(' '.join(news))

pos = WordCloud(
    background_color='black',
    max_words=100,
    max_font_size=60, 
    scale=20,
    random_state=42,
    collocations=False,
    normalize_plurals=False
).generate(' '.join(pos))

neutral = WordCloud(
    background_color='black',
    max_words=100,
    max_font_size=60, 
    scale=20,
    random_state=42,
    collocations=False,
    normalize_plurals=False
).generate(' '.join(neutral))

neg = WordCloud(
    background_color='black',
    max_words=100,
    max_font_size=60, 
    scale=20,
    random_state=42,
    collocations=False,
    normalize_plurals=False
).generate(' '.join(neg))

##Creating individual wordclouds per sentiment
fig, axs = plt.subplots(2, 2, figsize = (20, 12))
fig.tight_layout(pad = 0)

axs[0, 0].imshow(news)
axs[0, 0].set_title('News', fontsize = 20)
axs[0, 0].axis('off')

axs[0, 1].imshow(pos)
axs[0, 1].set_title('Positive ', fontsize = 20)
axs[0, 1].axis('off')

axs[1, 0].imshow(neg)
axs[1, 0].set_title('Negative ', fontsize = 20)
axs[1, 0].axis('off')

axs[1, 1].imshow(neutral)
axs[1, 1].set_title('Neutral  ', fontsize = 20)
axs[1, 1].axis('off')

plt.savefig('joint_cloud.png')
# Adding select words to stop words for better analysis on important word frequency
stop = set(stopwords.words('english')) 
stop_words = ["via", "co", "I",'We','The','going'] + list(stop)

# Removing stop words from the tweets
train_df['word'] = train_df['word_cloud'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))
train_df['word'] = train_df['word'].str.replace(r'[^\w\s]+', '')

# Separating the strings to a list of words
word_list = [word for line in train_df['word'] for word in line.split()]

# Creating a word frequency counter
sns.set(style="darkgrid")
counts = Counter(word_list).most_common(15)
counts_df = pd.DataFrame(counts)
counts_df
counts_df.columns = ['word', 'frequency']

# Creating a word frequency plot
fig, ax = plt.subplots(figsize = (9, 9))
ax = sns.barplot(y="word", x='frequency', ax = ax, data=counts_df, palette="hls")
plt.title('WORD FREQUENCY')
plt.savefig('wordcount_bar.png')
# Make a set of stop words
fig.suptitle('Bigrams in Tweets')
stopwords = set(STOPWORDS)
more_stopwords = {'https', 'amp','https rt'}
stopwords = stopwords.union(more_stopwords)

# Plot for each sentiment of bigrams
plt.figure(figsize=(16,12))

plt.subplot(2,2,1)
bigram_d = list(
    bigrams(
        [w for w in word_tokenize(' '.join(train_df.loc[train_df.sentiment==1, 'word']).lower()) 
        if (w not in stopwords) & (w.isalpha())]
    )
)

d_fq = FreqDist(bg for bg in bigram_d)
bgdf_d = pd.DataFrame.from_dict(d_fq, orient='index', columns=['count'])
bgdf_d.index = bgdf_d.index.map(lambda x: ' '.join(x))
bgdf_d = bgdf_d.sort_values('count',ascending=False)
sns.barplot(bgdf_d.head(10)['count'], bgdf_d.index[:10], color='pink')
plt.title('Positive Tweets')

plt.subplot(2,2,2)
bigram_nd = list(bigrams([w for w in word_tokenize(' '.join(train_df.loc[train_df.sentiment==2, 'word']).lower()) if 
              (w not in stopwords) & (w.isalpha())]))
nd_fq = FreqDist(bg for bg in bigram_nd)
bgdf_nd = pd.DataFrame.from_dict(nd_fq, orient='index', columns=['count'])
bgdf_nd.index = bgdf_nd.index.map(lambda x: ' '.join(x))
bgdf_nd = bgdf_nd.sort_values('count',ascending=False)
sns.barplot(bgdf_nd.head(10)['count'], bgdf_nd.index[:10], color='b')
plt.title('News Tweets')

plt.subplot(2,2,3)
bigram_nd = list(bigrams([w for w in word_tokenize(' '.join(train_df.loc[train_df.sentiment==-1, 'word']).lower()) if 
              (w not in stopwords) & (w.isalpha())]))
nd_fq = FreqDist(bg for bg in bigram_nd)
bgdf_nd = pd.DataFrame.from_dict(nd_fq, orient='index', columns=['count'])
bgdf_nd.index = bgdf_nd.index.map(lambda x: ' '.join(x))
bgdf_nd = bgdf_nd.sort_values('count',ascending=False)
sns.barplot(bgdf_nd.head(10)['count'], bgdf_nd.index[:10], color='c')
plt.title('Negative Tweets')

plt.subplot(2,2,4)
bigram_nd = list(bigrams([w for w in word_tokenize(' '.join(train_df.loc[train_df.sentiment==0, 'word']).lower()) if 
              (w not in stopwords) & (w.isalpha())]))
nd_fq = FreqDist(bg for bg in bigram_nd)
bgdf_nd = pd.DataFrame.from_dict(nd_fq, orient='index', columns=['count'])
bgdf_nd.index = bgdf_nd.index.map(lambda x: ' '.join(x))
bgdf_nd = bgdf_nd.sort_values('count',ascending=False)
sns.barplot(bgdf_nd.head(10)['count'], bgdf_nd.index[:10], color='g')
plt.title('Neutral Tweets')
plt.show()
def clean_tweets(message):
    """
    Cleaning all tweets by removing contractions, url-links, punctuation, digits,
    stopwords and Lemmatizing all the words.

    Returns
      A clean tweet as string
    """

    # change all words into lower case
    message = message.lower()

    #removing contractions
    message = contractions.fix(message)

    # replace all url-links with url-web
    url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    web = 'url-web'
    message = re.sub(url, web, message)

    # removing all punctuation and digits
    message = re.sub(r'[-]',' ',message)
    message = re.sub(r'[^\w\s]','',message)
    message = re.sub('[0-9]+', '', message)
    
    # removing stopwords
    nltk_stopword = nltk.corpus.stopwords.words('english')
    message = ' '.join([item for item in message.split() if item not in nltk_stopword])

    # lemmatizing all words
    message = message.lower()
    lemmatizer = WordNetLemmatizer()
    message = [lemmatizer.lemmatize(token) for token in message.split(" ")]
    message = [lemmatizer.lemmatize(token, "v") for token in message]
    message = " ".join(message)

    return message
train_df['message_clean']=train_df['message'].apply(clean_tweets)
test_df['message_clean']=test_df['message'].apply(clean_tweets)
duplicates_df2 = train_df[train_df['message_clean'].duplicated()]
duplicates_df2['sentiment'].value_counts()
sns.countplot(x = 'sentiment', data = duplicates_df2, palette="hls")
plt.title('Duplicated Tweets per Sentiment');
# Grouping duplicates dataframe and doing count of unique tweets
duplicates_df2= (
    duplicates_df2.groupby(['message_clean', 'sentiment'])
      .message.agg('count')
      .to_frame('Unique RT')
      .sort_values('Unique RT', ascending = False)
)
duplicates_df2 = duplicates_df2.reset_index(level=['message_clean','sentiment'])
duplicates_df2.head()
# Defining Features & Labels
X=train_df['message_clean']
y=train_df['sentiment']

#Having a split of 95%/5% yielded the best results.
X_train ,X_test ,y_train ,y_test = train_test_split(X,y,test_size =0.05, random_state =42)
#List of all models
classifiers = [
               LinearSVC(),
               svm.SVC(),
               tree.DecisionTreeClassifier(),
               RandomForestClassifier(n_estimators=100, 
                               max_depth=2, 
                               random_state=0, 
                               class_weight="balanced"),
               MLPClassifier(alpha=1e-5, 
                             hidden_layer_sizes=(5, 2), 
                             random_state=42),
               LogisticRegression(random_state=123, 
                                  multi_class='ovr',
                                  n_jobs=1, 
                                  C=1e5,
                                  max_iter = 4000),
               KNeighborsClassifier(n_neighbors=3),
               MultinomialNB(),
               ComplementNB(),
               SGDClassifier(loss='hinge', 
                             penalty='l2',
                             alpha=1e-3, 
                             random_state=42, 
                             max_iter=5, 
                             tol=None),
               GradientBoostingClassifier(),
               xgboost.XGBClassifier(learning_rate =0.1,
                                     n_estimators=1000,
                                     max_depth=5, 
                                     min_child_weight=1,
                                     gamma=0,
                                     subsample=0.8,
                                     colsample_bytree=0.8,
                                     nthread=4,
                                     seed=27)
    ]
def model_building(classifiers, X_train, y_train, X_test,y_test):
    """Function to build a variety of classifiers and return a summary of F1-score
    and processing time as a dataframe
    """ 
    model_summary = {}
    
    # Pipeline to balance the classses and then to build the model
    for clf in classifiers:
      text_clf = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', 
                             min_df=1, 
                             max_df=0.9, 
                             ngram_range=(1, 2))),
            ('clf',clf)
      ])

      # Logging the Execution Time for each model
      start_time = time.time()
      text_clf.fit(X_train, y_train)
      predictions = text_clf.predict(X_test)
      run_time = time.time()-start_time
      
      # Output for each model: F1_Macro, F1_Accuracy, F1_Weighted & Execution TIme
      model_summary[clf.__class__.__name__] = {
          'F1-Macro':metrics.f1_score(y_test,predictions,average='macro'),
          'F1-Accuracy':metrics.f1_score(y_test,predictions,average='micro'),
          'F1-Weighted':metrics.f1_score(y_test,predictions,average='weighted'),
          'Execution Time': run_time
      }
        
    return pd.DataFrame.from_dict(model_summary, orient='index')
classifiers_df = model_building(classifiers,X_train, y_train, X_test, y_test)
ordered_df = classifiers_df.sort_values('F1-Macro',ascending=False)
ordered_df
ordered_df = ordered_df.rename_axis(index='Model')
ordered_df = ordered_df.reset_index(level='Model')
ax = plt.gca()

ordered_df.plot(kind='line',x='Model',y='F1-Macro',ax=ax)
ordered_df.plot(kind='line',x='Model',y='F1-Accuracy', color='red', ax=ax)
plt.xticks(rotation=90)
plt.show()
ax = plt.gca()
ordered_df.plot(kind='bar',x='Model',y='Execution Time',ax=ax)
plt.xticks(rotation=90)
plt.show()
def cross_val_models(X,y):
    """Function to build a variety of classifiers and return a summary of F1-score
    and processing time as a dataframe
    """ 
    model_summary = []
    
    for clf in classifiers:
      if clf.__class__.__name__ == 'XGBClassifier':
        continue
      text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(
          stop_words='english', 
          min_df=1, 
          max_df=0.9, 
          ngram_range=(1, 2))
        ),
        ('clf',clf)
      ])

      start_time = time.time()
      scores = cross_val_score(text_clf, X=X, y=y, cv=10)
      run_time = time.time()-start_time
      model_summary.append([clf.__class__.__name__, scores.mean(), scores.std(), run_time ])
    
    cv = pd.DataFrame(model_summary, columns=['Model', 'CV_Mean', 'CV_Std_Dev', 'Execution Time'])
    cv.set_index('Model', inplace=True)
      
    return cv
cross_val_df = cross_val_models(X,y)
cross_val_df = cross_val_df.sort_values('CV_Mean',ascending=False)
cross_val_df
cross_val = cross_val_df.reset_index()
ax = plt.gca()
cross_val.plot(kind='line',x='Model',y='CV_Mean',ax=ax)
plt.xticks(rotation=90)
plt.show()
ax = plt.gca()
cross_val.plot(kind='bar',x='Model',y='Execution Time', color='blue', ax=ax)
plt.xticks(rotation=90)
plt.show()
tfidf =TfidfVectorizer(stop_words='english',min_df=1, 
                             max_df=0.9, 
                             ngram_range=(1, 2))

X_Tfidf =tfidf.fit_transform(train_df['message_clean'])
# Redefining X and y variables after Vectorization
X=X_Tfidf
y=train_df['sentiment']
X_train ,X_test ,y_train ,y_test = train_test_split(X,y,test_size =0.05, random_state =42)
# API key to run experiment in Comet
# experiment = Experiment(api_key="h9aq14TfOuTPJxNhr12fk20kk",
#                         project_name="tweet-classification", 
#                         workspace="maddy-muir")

param_grid = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01], 
              'kernel': ['linear']}

SVC_grid = GridSearchCV(SVC(), 
                        param_grid, 
                        refit = True, 
                        verbose = 3, 
                        scoring = 'f1_macro')
  
# fitting the model for grid search 
SVC_grid.fit(X_train, y_train) 
y_pred = SVC_grid.predict(X_test)

f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
accuracy = metrics.f1_score(y_test, y_pred, average='micro')

# experiment.log_dataset_hash(X_train)
# experiment.log_parameters({"model_type": "Linear SVC", "param_grid": param_grid})
# experiment.log_metrics({'F1 Macro': f1_macro, "Accuracy": accuracy})

# experiment.end()
print(SVC_grid.best_score_)
print(SVC_grid.best_params_) 
print(SVC_grid.best_estimator_) 
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
#  API key to run experiment in Comet
# experiment = Experiment(api_key="h9aq14TfOuTPJxNhr12fk20kk",
#                         project_name="tweet-classification", 
#                         workspace="maddy-muir")

# define models and parameters
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
LR_model = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='f1_macro',error_score=0)
LR_model.fit(X_train, y_train)

y_pred = LR_model.predict(X_test)
f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
accuracy = metrics.f1_score(y_test, y_pred, average='micro')

# experiment.log_dataset_hash(X_train)
# experiment.log_parameters({"model_type": "Logistic Regression", "param_grid": grid})
# experiment.log_metrics({'F1 Macro': f1_macro, "Accuracy": accuracy})

# experiment.end()
print(LR_model.best_score_)
print(LR_model.best_params_) 
print(LR_model.best_estimator_) 
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
# API key to run experiment in Comet
# experiment = Experiment(api_key="h9aq14TfOuTPJxNhr12fk20kk",
#                         project_name="tweet-classification", 
#                         workspace="maddy-muir")

model = RidgeClassifier()
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# define grid search
grid = dict(alpha=alpha)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
RC_model = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
RC_model.fit(X_train, y_train)

y_pred = RC_model.predict(X_test)
f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
accuracy = metrics.f1_score(y_test, y_pred, average='micro')

# experiment.log_dataset_hash(X_train)
# experiment.log_parameters({"model_type": "Ridge Classifier", "param_grid": grid})
# experiment.log_metrics({'F1 Macro': f1_macro, "Accuracy": accuracy})

# experiment.end()
print(RC_model.best_score_)
print(RC_model.best_params_) 
print(RC_model.best_estimator_) 
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
# API key to run experiment in Comet
# experiment = Experiment(api_key="h9aq14TfOuTPJxNhr12fk20kk",
#                         project_name="tweet-classification", 
#                         workspace="maddy-muir")

model = KNeighborsClassifier()
n_neighbors = range(1, 21, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

# define grid search
grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
KN_model = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
KN_model.fit(X_train, y_train)

y_pred = KN_model.predict(X_test)
f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
accuracy = metrics.f1_score(y_test, y_pred, average='micro')

# experiment.log_dataset_hash(X_train)
# experiment.log_parameters({"model_type": "KNeighbours", "param_grid": grid})
# experiment.log_metrics({'F1 Macro': f1_macro, "Accuracy": accuracy})

# experiment.end()
print(KN_model.best_score_)
print(KN_model.best_params_) 
print(KN_model.best_estimator_) 
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
# API key to run experiment in Comet
# experiment = Experiment(api_key="h9aq14TfOuTPJxNhr12fk20kk",
#                         project_name="tweet-classification", 
#                         workspace="maddy-muir")

model = ComplementNB()
alpha =[0.01, 0.1, 0.5, 1, 10]
fit_prior = [True, False]
norm = [True, False]

grid = dict(alpha=alpha, fit_prior=fit_prior, norm=norm)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
NB_model = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
NB_model.fit(X_train,y_train)

y_pred = NB_model.predict(X_test)
f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
accuracy = metrics.f1_score(y_test, y_pred, average='micro')

# experiment.log_dataset_hash(X_train)
# experiment.log_parameters({"model_type": "Complement NB", "param_grid": grid})
# experiment.log_metrics({'F1 Macro': f1_macro, "Accuracy": accuracy})

# experiment.end()
print(NB_model.best_score_)
print(NB_model.best_params_) 
print(NB_model.best_estimator_) 
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
#List of all models
final_models = [
               SVC(C=10, gamma=1, kernel='linear'),
               LinearSVC(),
               LogisticRegression(C=100, 
                                  class_weight=None, 
                                  dual=False,
                                  fit_intercept=True,
                                  intercept_scaling=1, 
                                  l1_ratio=None,
                                  max_iter=100, 
                                  multi_class='auto', 
                                  n_jobs=None,
                                  penalty='l2',
                                  random_state=None, 
                                  solver='lbfgs',
                                  tol=0.0001,
                                  warm_start=False),
               KNeighborsClassifier(metric='euclidean', 
                                  n_neighbors=15, 
                                  weights='distance'),
               ComplementNB(alpha=1, 
                            class_prior=None, 
                            fit_prior=True, 
                            norm=True),
               RidgeClassifier(alpha=0.5)    
    ]
X=train_df['message_clean']
y=train_df['sentiment']
X_train ,X_test ,y_train ,y_test = train_test_split(X,y,test_size =0.05, random_state =42)
final_check = model_building(final_models, X_train, y_train, X_test, y_test)
final_ordered = final_check.sort_values('F1-Macro',ascending=False)
final_ordered
def final_model_fitting(classifiers, X, y):
    """Function to build all the final classifiers 
    """ 
    model_summary = {}
    
    for clf in classifiers:
        text_clf = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', 
                             min_df=1, 
                             max_df=0.9, 
                             ngram_range=(1, 2))),
            ('clf',clf)
        ])
    
        text_clf.fit(X,y)

        model_summary[clf.__class__.__name__] = text_clf
      
    return model_summary
dict_final = final_model_fitting(final_models,X,y);
final_ordered = final_ordered.rename_axis(index='Model')
final_ordered = final_ordered.reset_index(level='Model')

ax = plt.gca()

final_ordered.plot(kind='line',x='Model',y='F1-Macro',ax=ax)
final_ordered.plot(kind='line',x='Model',y='F1-Accuracy', color='red', ax=ax)
plt.xticks(rotation=90)
plt.show()
ax = plt.gca()
final_ordered.plot(kind='bar',x='Model',y='Execution Time', color='blue', ax=ax)
plt.show()
# Download CSV file for each one of the final models
for key, model in dict_final.items():
  test_df['sentiment'] = model.predict(test_df['message_clean'])
  submission = test_df[['tweetid', 'sentiment']]
  submission.to_csv(f'{key}.csv',index=False)
# Function to Pickle model for use within  API
def save_pickle(filename, model):
    """Pickle model for use within our API"""
    save_path = f'{filename}.pkl'
    print (f"Training completed. Saving model to: {save_path}")
    pickle.dump(model, open(save_path,'wb'))
import pickle
for key, model in dict_final.items():
  save_pickle(key,model)
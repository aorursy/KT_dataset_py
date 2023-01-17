import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import random as rn
import seaborn as sns
import tensorflow as tf
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt  
from urllib.parse import urlparse
import plotly as ply
#clean data
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not", "can't" : "cannot", "couldn't" : "could not", "couldnt" : "could not", "didn't" : "did not", "doesn't" : "does not",
                "doesnt" : "does not", "don't" : "do not", "hadn't" : "had not", "hasn't" : "has not", "haven't" : "have not", "havent" : "have not",
                "he'd" : "he would", "he'll" : "he will", "he's" : "he is", "i'd" : "I would", "i'd" : "I had", "i'll" : "I will", "i'm" : "I am",
                "isn't" : "is not","it's" : "it is","it'll":"it will", "i've" : "I have", "let's" : "let us", "mightn't" : "might not", "mustn't" : "must not",
                "shan't" : "shall not", "she'd" : "she would", "she'll" : "she will", "she's" : "she is", "shouldn't" : "should not", "shouldnt" : "should not",
                "that's" : "that is", "thats" : "that is", "there's" : "there is", "theres" : "there is", "they'd" : "they would", "they'll" : "they will",
                "they're" : "they are", "theyre":  "they are", "they've" : "they have", "we'd" : "we would", "we're" : "we are", "weren't" : "were not",
                "we've" : "we have", "what'll" : "what will", "what're" : "what are", "what's" : "what is", "what've" : "what have", "where's" : "where is",
                "who'd" : "who would", "who'll" : "who will", "who're" : "who are", "who's" : "who is", "who've" : "who have", "won't" : "will not",
                "wouldn't" : "would not", "you'd" : "you would", "you'll" : "you will", "you're" : "you are", "you've" : "you have", "'re": " are",
                "wasn't": "was not", "we'll":" will", "didn't": "did not", "tryin'":"trying"}

def clean_text(x):
    x = str(x).replace("\n","")
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

def clean_data(df, columns):
    for col in tqdm(columns):
        df[col] = df[col].apply(lambda x: re.sub(' +', ' ', x)).values
        df[col] = df[col].apply(lambda x: re.sub('\n', '', x)).values
        df[col] = df[col].apply(lambda x: clean_numbers(x)).values
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x)).values
        df[col] = df[col].apply(lambda x: clean_text(x.lower())).values
        df[col] = df[col].apply(lambda x: x.lower()).values
        df[col] = df[col].apply(lambda x: re.sub(' +', ' ', x)).values

    return df
from sklearn.preprocessing import MinMaxScaler
def preprocess_data(train):

  y = train[train.columns[11:]] # storing the target labels in 'y'

  # I'll be cleaning and adding the domain name from the website's url.
  find = re.compile(r"^[^.]*")
  train['clean_url'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

  # creating train and test data
  X = train[['question_title', 'question_body', 'answer', 'host', 'category']]
  text_features = ['question_title', 'question_body', 'answer']

  # Cleaning data for contracted words, numbers and punctuations.
  X = clean_data(X, text_features)

  return X
X = pd.read_csv('../input/google-quest-challenge/train.csv').iloc[:, 11:] 
unique_labels = np.unique(X.values)
denominator = 60
q = np.arange(0, 101, 100 / denominator)
exp_labels = np.percentile(unique_labels, q) # Generating the 60 bins.

def optimize_ranks(preds, unique_labels=exp_labels): 
  new_preds = np.zeros(preds.shape)
  for i in range(preds.shape[1]):
    interpolate_bins = np.digitize(preds[:, i], bins=unique_labels, right=False)-1
    if len(np.unique(interpolate_bins)) == 1:
      new_preds[:, i] = preds[:, i]
    else:
      # new_preds[:, i] = unique_labels[interpolate_bins]
      new_preds[:, i] = interpolate_bins
  
  return new_preds
y_true = pd.read_csv('../input/google-quest-challenge/train.csv').iloc[:, 11:]
y_pred = pd.read_csv('../input/google-quest-qna-bert-pred/pred_train.csv').iloc[:, 1:]
y_true = optimize_ranks(y_true.values)
y_pred = optimize_ranks(y_pred.values)
# Generating the MSE-score for each data point in train data.
from sklearn.metrics import mean_squared_error
train_score = [mean_squared_error(i,j) for i,j in zip(y_pred, y_true)]
# sorting the losses from minimum to maximum imdex wise.
train_score_args = np.argsort(train_score)
train = pd.read_csv('../input/google-quest-challenge/train.csv')
X_train = preprocess_data(train)
# function for generating wordcloud
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
sns.set()

def generate_wordcloud(indexes, data, color='black'):
  comment_words = '' 
  stopwords = set(STOPWORDS)

  title_words = data['question_title'].iloc[i]
  body_words = data['question_body'].iloc[i]
  answer_words = data['answer'].iloc[i]

  title_cloud = WordCloud(width = 400, height = 200, background_color = color,
                        stopwords = stopwords, min_font_size = 10).generate(title_words)

  body_cloud = WordCloud(width = 400, height = 200, background_color = color,
                        stopwords = stopwords, min_font_size = 10).generate(body_words)

  answer_cloud = WordCloud(width = 400, height = 200, background_color = color,
                        stopwords = stopwords, min_font_size = 10).generate(answer_words)
  
  return title_cloud, body_cloud, answer_cloud
# I've picked the top 5 datapoints from train data with lowest loss and plotted the wordcloud of their question_title, question_body and answer.
print('Top 5 data points from train data that give the "lowest" loss.')
for i, idx in enumerate(train_score_args[:5]):
  title, body, answer = generate_wordcloud(idx, X_train)
  plt.figure(figsize=(20,12))
  plt.subplot(131)
  plt.imshow(title)
  if i==0: plt.title('question_title')
  plt.ylabel(f'loss: {train_score[idx]}')
  plt.subplot(132)
  plt.imshow(body)
  if i==0: plt.title('question_body')
  plt.subplot(133)
  plt.imshow(answer)
  if i==0: plt.title('answer')
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
  plt.show()
# I've picked the top 5 datapoints from train data with 'highest' loss and plotted the wordcloud of their question_title, question_body and answer.
print('Top 5 data points from Train data that give the "highest" loss.')
for i, idx in enumerate(train_score_args[-5:]):
  title, body, answer = generate_wordcloud(idx, X_train, color='white')
  plt.figure(figsize=(20,12))
  plt.subplot(131)
  plt.imshow(title)
  if i==0: plt.title('question_title')
  plt.ylabel(f'loss: {train_score[idx]}')
  plt.subplot(132)
  plt.imshow(body)
  if i==0: plt.title('question_body')
  plt.subplot(133)
  plt.imshow(answer)
  if i==0: plt.title('answer')
  plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
  plt.show()

# I've picked the top 30 datapoints from train and cv data with 'lowest' loss and plotted the word counts of their question_title, question_body and answer.
print("word counts of the question_title, question_body and answer of top 30 train and cv data with 'lowest' loss.")
i = 30
title_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[:i]]['question_title'].values]
body_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[:i]]['question_body'].values]
answer_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[:i]]['answer'].values]

plt.figure(figsize=(20,4))
plt.subplot(131)
plt.plot(title_train_len)
plt.title('question_title (train data)')
plt.ylabel('number of words')
plt.xlabel('datapoint (loss: high --> low)')
plt.subplot(132)
plt.plot(body_train_len)
plt.title('question_body (train data)')
plt.xlabel('datapoint (loss: low --> high)')
plt.subplot(133)
plt.plot(answer_train_len)
plt.title('answer (train data)')
plt.xlabel('datapoint (loss: high --> low)')
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.show()
# I've picked the top 30 datapoints from train and cv data with 'highest' loss and plotted the word counts of their question_title, question_body and answer.
print("word counts of the question_title, question_body and answer of top 30 train and cv data with 'highest' loss.")
i = -30
title_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[i:]]['question_title'].values]
body_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[i:]]['question_body'].values]
answer_train_len = [len(l.split(' ')) for l in X_train.iloc[train_score_args[i:]]['answer'].values]

plt.figure(figsize=(20,4))
plt.subplot(131)
plt.plot(title_train_len)
plt.title('question_title (train data)')
plt.ylabel('number of words')
plt.xlabel('datapoint (loss: high --> low)')
plt.subplot(132)
plt.plot(body_train_len)
plt.title('question_body (train data)')
plt.xlabel('datapoint (loss: high --> low)')
plt.subplot(133)
plt.plot(answer_train_len)
plt.title('answer (train data)')
plt.xlabel('datapoint (loss: high --> low)')
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.show()

# I've picked the top 100 datapoints from train data with 'highest' loss and collected the values of domain names.
top_url = X_train['host'].iloc[train_score_args[:100]].value_counts()
bottom_url = X_train['host'].iloc[train_score_args[-100:]].value_counts()
# Top 10 frequently occuring domain names that lead to minimum loss
top_url[1:10].plot.bar(figsize=(12,8))
plt.title('top 10 url domain that produce the minimum loss')
plt.ylabel('frequency')
plt.show()
# Top 10 frequently occuring domain names that lead to maximum loss
bottom_url[1:10].plot.bar(figsize=(12,8))
plt.title('top 10 url domain that produce the maximum loss')
plt.ylabel('frequency')
plt.show()
# finding the unique domain names that contribute to low and high losses
best_url = ' '.join(list(set(top_url.keys()) - set(bottom_url.keys()))) # set of urls that contribute solely to low loss
worst_url = ' '.join(list(set(bottom_url.keys()) - set(top_url.keys()))) # set of urls that contribute solely to high loss
best_url_cloud = WordCloud(width = 400, height = 200, background_color ='orange',
                           stopwords = STOPWORDS, min_font_size = 10).generate(best_url)

worst_url_cloud = WordCloud(width = 400, height = 200, background_color ='cyan',
                            stopwords = STOPWORDS, min_font_size = 10).generate(worst_url)

plt.figure(figsize=(20,12))
plt.subplot(121)
plt.imshow(best_url_cloud)
plt.title('url domain with well predicted labels (low loss)')
plt.subplot(122)
plt.imshow(worst_url_cloud)
plt.title('url domain with bad predicted labels (high loss)')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.show()
# for train data
plt.figure(figsize=(20,20))
plt.subplot(121)
X_train['category'].iloc[train_score_args[:100]].value_counts().plot.pie(autopct='%1.1f%%', explode=(0,0.02,0.04,0.06,0.08), shadow=True)
plt.ylabel('')
plt.title('categories of best fitted data points with minimum loss (on train data)')
plt.subplot(122)
X_train['category'].iloc[train_score_args[-100:]].value_counts().plot.pie(autopct='%1.1f%%', explode=(0,0.02,0.04,0.06,0.08), shadow=True)
plt.ylabel('')
plt.title('categories of worst fitted data points with maximum loss (on train data)')
plt.show()


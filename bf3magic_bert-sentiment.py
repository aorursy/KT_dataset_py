# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D, Bidirectional
from sklearn.model_selection import train_test_split
from keras.initializers import Constant
from keras.optimizers import Adam
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATASET_ENCODING = "ISO-8859-1"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_PATH = "../input/sentiment140/training.1600000.processed.noemoticon.csv"
CURSE_WORDS = ["asshole", "bitch", "crap", "cunt", "damn", "fuck", "hell", "shit", "slut", "nigga", "prick"]
twitter = pd.read_csv(DATASET_PATH, encoding = DATASET_ENCODING, names = DATASET_COLUMNS)
twitter.head(3)
twitter = twitter[["target", "text"]]
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")

def process_text(text):
  text = hashtags.sub(' hashtag', text)
  text = mentions.sub(' entity', text)
  return text.strip().lower()
  
def match_expr(pattern, string):
  return not pattern.search(string) == None

def get_data_wo_urls(dataset):
    link_with_urls = dataset.text.apply(lambda x: match_expr(urls, x))
    return dataset[[not e for e in link_with_urls]]
twitter['text'] = twitter['text'].apply(lambda x : remove_URL(x))
twitter['text'] = twitter['text'].apply(lambda x : remove_html(x))
twitter['text'] = twitter['text'].apply(lambda x : remove_emoji(x))
twitter['text'] = twitter['text'].apply(lambda x : remove_punct(x))
twitter.text = twitter.text.apply(process_text)

curse_words = ' ' + ' | '.join(CURSE_WORDS) + ' '
twitter['text'] = twitter['text'].apply(lambda x : ' ' + x + ' ')
curse_tweets = twitter[twitter["text"].str.contains(curse_words, case=False)]
curse_tweets.shape

curse_tweets['target'] = curse_tweets['target'].apply(lambda x : 1 if x == 4 else x)
x = curse_tweets.target.value_counts()
sns.barplot(x.index, x)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len=curse_tweets[curse_tweets['target']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='red')
ax1.set_title('positive tweets')
tweet_len=curse_tweets[curse_tweets['target']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='green')
ax2.set_title('negative tweets')
fig.suptitle('Words in a tweet with curse words')
plt.show()
!pip install bert-for-tf2

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 1 - TRAIN_SIZE - VAL_SIZE

train_val, test = train_test_split(curse_tweets, test_size = TEST_SIZE, random_state = 42)
print("TRAIN size: ", len(train_val))
print("TEST size: ", len(test))

!wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip uncased_L-12_H-768_A-12.zip
os.makedirs("./model", exist_ok=True)
!mv uncased_L-12_H-768_A-12/ ./model
bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir = os.path.join("./model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")
tokenizer = FullTokenizer(vocab_file = "./model/uncased_L-12_H-768_A-12/vocab.txt")
def create_model(max_seq_len, bert_ckpt_file):

  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = None
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name="input_ids")
  bert_output = bert(input_ids)

  print("bert shape", bert_output.shape)

  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
  cls_out = keras.layers.Dropout(0.7)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.7)(logits)
  logits = keras.layers.Dense(1, activation="sigmoid")(logits)

  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  load_stock_weights(bert, bert_ckpt_file)
        
  return model
    
train_val.head()
class SentimentAnalysisData:
  DATA_COLUMN = "text"
  LABEL_COLUMN = "target"

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes
    
    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []
    
    for _, row in tqdm(df.iterrows()):
      text, label = row[SentimentAnalysisData.DATA_COLUMN], row[SentimentAnalysisData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))

    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)
classes = train_val.target.unique().tolist()

data = SentimentAnalysisData(train_val, test, tokenizer, classes, max_seq_len=40)
data.train_x.shape
print(len(data.train_y[data.train_y==0]))
print(len(data.train_y[data.train_y==1]))
print(len(train_val[train_val["target"]==0]))
print(len(train_val[train_val["target"]==1]))
print(len(test[test["target"]==0]))
print(len(test[test["target"]==1]))
data.train_x[0]

data.train_y[0]
data.max_seq_len
import tensorflow as tf
import keras

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
with tpu_strategy.scope():
    model = create_model(data.max_seq_len, bert_ckpt_file)
    model.summary()
    model.compile(
      optimizer=keras.optimizers.Adam(1e-5),
      loss="binary_crossentropy",
      metrics=["accuracy"]
    )

history = model.fit(
  x=data.train_x,
  y=data.train_y,
  validation_split=0.2,
  batch_size=32,
  shuffle=True,
  epochs=5,
  verbose=1
)
_, test_acc = model.evaluate(data.test_x, data.test_y)
_, train_acc = model.evaluate(data.train_x, data.train_y)
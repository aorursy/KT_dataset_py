import numpy as np
import pandas as pd

#Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import plotly.offline
import plotly.express as px
import plotly.graph_objects as go

#Natural Language Processing
#Data Manipulation and Cleaning
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import Counter
stop = set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import gensim
import string

#Modeling
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from keras.initializers import Constant
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train = Train.copy()
train.head()
print('There are {} rows and {} columns in train'.format(train.shape[0], train.shape[1]))
#Category counts for type of tweets
Category_count=np.array(train['target'].value_counts())
Tweet_type=sorted(train['target'].unique())
fig = go.Figure(data=[go.Pie(labels=Tweet_type, values=Category_count, hole=.3)])
fig.show()
train['target'].value_counts()
#Adding lenght column to dataset
train['length']=train['text'].apply(len)
train.head()
#checking length distribution
import plotly.express as px
fig = px.histogram(train, x="length", color="target")
fig.show()
train['word_count']=train['text'].str.split().map(lambda x: len(x))
import plotly.express as px
fig = px.histogram(train, x="word_count", color="target")
fig.show()
train.head()
train.describe()
#Tweet with max length
train[train['length']==157]['text'].iloc[0]
train[train['length']==7]['text'].iloc[0]
#Tweet with max word count
train[train['word_count']==31]['text'].iloc[0]
train[train['word_count']==1]['text'].iloc[0]
avg_word_length=train['text'].str.split().apply(lambda x : [len(i) for i in x])
train['avg_word_length']=avg_word_length.map(lambda x: np.mean(x))
train.head()
import plotly.express as px
fig = px.histogram(train, x="avg_word_length", color="target")
fig.show()
#Creating Tweet Corpus function
def create_corpus(target):
    corpus=[]
    
    for x in train[train['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
            
    return corpus        
corpus = create_corpus(1)
dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top = sorted(dic.items(), key = lambda x:x[1], reverse = True)[:10]        
x,y = zip(*top)
plt.bar(x,y, color = 'pink')
corpus = create_corpus(0)
dic = defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top = sorted(dic.items(), key = lambda x:x[1], reverse = True)[:10]        
x,y = zip(*top)
plt.bar(x,y, color = 'pink')
plt.figure(figsize = (10,5))
corpus = create_corpus(1)

dic = defaultdict(int)
import string
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i] += 1
        
x, y = zip(*dic.items())
plt.bar(x, y, color='purple')
        
plt.figure(figsize = (10,5))
corpus = create_corpus(0)

dic = defaultdict(int)
import string
special = string.punctuation
for i in (corpus):
    if i in special:
        dic[i] += 1
        
x, y = zip(*dic.items())
plt.bar(x, y, color = 'purple')
        
corpus = create_corpus(1)
counter = Counter(corpus)
most = counter.most_common()
x = []
y = []
for word, count in most[:40]:
    if (word not in stop):
        x.append(word)
        y.append(count)
sns.barplot(x=y,y=x)
corpus = create_corpus(0)
counter = Counter(corpus)
most = counter.most_common()
x = []
y = []
for word, count in most[:40]:
    if (word not in stop):
        x.append(word)
        y.append(count)
sns.barplot(x=y,y=x)
def get_top_tweet_bigrams(corpus, n = None):
    vec = CountVectorizer(ngram_range = (2,2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]
    
plt.figure(figsize = (10,5))
top_tweet_bigrams = get_top_tweet_bigrams(train['text'])[:10]
x,y = map(list, zip(*top_tweet_bigrams))
sns.barplot(x=y, y=x)
df = pd.concat([Train,test])
df.shape
example="New competition launched :https://www.kaggle.com/c/nlp-getting-started"
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
remove_URL(example)
df['text'] = df['text'].apply(lambda x : remove_URL(x))
example = """<div>
<h1>Real or Fake</h1>
<p>Kaggle </p>
<a href="https://www.kaggle.com/c/nlp-getting-started">getting started</a>
</div>"""
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'',text)
print(remove_html(example))
df['text']=df['text'].apply(lambda x : remove_html(x))
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

remove_emoji("Omg another Earthquake 😔😔")
df['text'] = df['text'].apply(lambda x : remove_emoji(x))
def remove_punct(text):
    table = str.maketrans('','', string.punctuation)
    return text.translate(table)

example = "I am a #king"
print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
def create_corpus(df):
    corpus = []
    for tweet in tqdm(df['text']):
        words = [word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus     
corpus = create_corpus(df)
embedding_dict = {}
with open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors
f.close()        
MAX_LEN = 50
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences = tokenizer_obj.texts_to_sequences(corpus)

tweet_pad = pad_sequences(sequences, maxlen = MAX_LEN, truncating = 'post', padding = 'post')
word_index = tokenizer_obj.word_index
print('Number of unique words:', len(word_index))
num_words = len(word_index)+1
embedding_matrix = np.zeros((num_words, 100))
for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
        
    emb_vec = embedding_dict.get(word)   
    if emb_vec is not None:
        embedding_matrix[i] = emb_vec
model=Sequential()

embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))


optimzer=Adam(learning_rate=1e-5)

model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])
model.summary()
train_ = tweet_pad[:train.shape[0]]
test = tweet_pad[train.shape[0]:]
X_train, X_test, y_train, y_test = train_test_split(train_,train['target'].values, test_size = 0.15)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_test.shape)
history = model.fit(X_train,y_train, batch_size = 4, epochs =15, validation_data = (X_test, y_test), verbose = 2)
sample_sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
y_pre=model.predict(test)
y_pre=np.round(y_pre).astype(int).reshape(3263)
sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})
sub.to_csv('submission.csv',index=False)
sub.head()

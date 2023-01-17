# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
train_df.head()
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test_df.head()
train_df.info()
test_df.info()
plt.style.use('ggplot')

import seaborn

x = train_df.target.value_counts()



seaborn.barplot(x.index,x)

plt.gca().set_ylabel('samples')
fig,(ax1,ax2) =plt.subplots(1,2,figsize =(10,5))

tweet_len = train_df[train_df['target'] ==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len,color='blue')

ax1.set_title('Disaster Tweets')

tweet_len =train_df[train_df['target'] ==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len,color='orange')

ax2.set_title('Normal Tweets')

fig.suptitle('Words of Tweets')

plt.show()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

word = train_df[train_df['target'] ==1]['text'].str.split().apply(lambda x: [len(i) for i in x])

seaborn.distplot(word.map(lambda x : np.mean(x)),ax = ax1,color='red')

ax1.set_title('Disaster Tweets')

word = train_df[train_df['target']==0]['text'].str.split().apply(lambda x:[len(i) for i in x])

seaborn.distplot(word.map(lambda x : np.mean(x)),ax = ax2,color ='blue')

ax2.set_title('Normal Tweets')

fig.suptitle('Average word length in each tweet')

plt.show()
from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter
def create_corpus(target):

    corpus =[]

    

    for x in train_df[train_df['target'] == target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus

    
stop=set(stopwords.words('english'))
corpus=create_corpus(0)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 





       
x,y= zip(*top)

plt.bar(x,y)
corpus = create_corpus(1)



dic=defaultdict(int)

for word in corpus:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

x,y = zip(*top)

plt.bar(x,y)
plt.figure(figsize=(10,5))

corpus = create_corpus(1)



dic = defaultdict(int)

import string

punct = string.punctuation



for i in corpus:

    if i in punct:

        dic[i]+=1



x,y =zip(*dic.items())

plt.bar(x,y)

plt.figure(figsize=(10,5))

corpus = create_corpus(0)



dic = defaultdict(int)

import string

punct = string.punctuation



for i in corpus:

    if i in punct:

        dic[i]+=1



x,y =zip(*dic.items())

plt.bar(x,y,color ='green')

counter = Counter(corpus)

most = counter.most_common()



x=[]

y=[]

for word,count in most[:50]:

    if (word not in stop):

        x.append(word)

        y.append(count)
seaborn.barplot(x=y,y=x) #x axis is count and y axis is word

plt.show()
def get_top_tweet_bigrams(corpus ,n = None):

    vec = CountVectorizer(ngram_range=(2,2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis =0)

    

    words_freq = [(word,sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]

    words_freq = sorted(words_freq,key = lambda x : x[1], reverse = True)

    

    return words_freq[:n]

    



    
plt.figure(figsize=(10,5))

top_tweet_bigrams = get_top_tweet_bigrams(train_df['text'])[:10]



x,y = map(list,zip(*top_tweet_bigrams))



seaborn.barplot(x=y,y=x)

plt.show()

df = pd.concat([train_df,test_df])

df.shape


import re



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



df['text'] = df['text'].apply(lambda x : remove_URL(x))
def remove_HTML(text):

    html = re.compile(r'<.*?>')

    return html.sub(r'',text)



    
df['text'] = df['text'].apply(lambda x:remove_HTML(x))
def remove_emojis(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'',text)





    

                       

                    
df['text'] = df['text'].apply(lambda x: remove_emojis(x))
#Reference

#https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
import string

def remove_punc(text):

    translator=str.maketrans('','',string.punctuation)

    return text.translate(translator)





    
df['text'] = df['text'].apply(lambda x: remove_punc(x))
!pip install pyspellchecker
from spellchecker import SpellChecker



spell = SpellChecker()



def correct_spelling(text):

    corrected_text=[]

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)



#df['text'] = df['text'].apply(lambda x: correct_spelling(x))
from nltk.tokenize import word_tokenize

import gensim



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

def create_corpus(df):

    corpus=[]

    for train_df in tqdm(df['text']):

        words = [word.lower() for word in word_tokenize(train_df) if ((word.isalpha() ==1) & (word not in stop))]

        corpus.append(words)

        

    return corpus

                                                                      

    
corpus = create_corpus(df)
embedding_dict={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN = 50

tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences = tokenizer_obj.texts_to_sequences(corpus)



tweet_pad = pad_sequences(sequences ,maxlen = MAX_LEN ,truncating = 'post',padding = 'post')

word_index = tokenizer_obj.word_index

print('Number of unique words :  ' ,len(word_index))
num_words = len(word_index) +1

embedding_matrix = np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

        

    emb_vec = embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i] = emb_vec

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
model = Sequential()

embedding = Embedding(num_words,100,embeddings_initializer = Constant(embedding_matrix),input_length = MAX_LEN,

                     trainable = False)

model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64,dropout = 0.2 ,recurrent_dropout = 0.2))

model.add(Dense(1,activation = 'sigmoid'))



optimizer = Adam(learning_rate  =1e-5)



model.compile(loss = 'binary_crossentropy',optimizer = optimizer , metrics = ['accuracy'])

          

          

          

          
model.summary()
train = tweet_pad[:train_df.shape[0]]

test = tweet_pad[train_df.shape[0]:]

X_train,X_test,y_train,y_test =train_test_split(train,train_df['target'].values,test_size = 0.2)

print('Shape of train: ' ,X_train.shape)

print('Shape of validation:', X_test.shape)
history = model.fit(X_train, y_train , batch_size = 4 ,epochs=15,validation_data = (X_test, y_test),verbose =2)
sample = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')



sample.head()
y_pred = model.predict(test)

y_pred = np.round(y_pred).astype(int).reshape(3263)



sub = pd.DataFrame({'id': sample['id'].values.tolist(),'target' : y_pred})



sub.to_csv('submission.csv',index = False)



sub.head()
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train.head()
test.head()
for col in train.columns:

    print("No of unique values in column --{} is --{}".format(col, train[col].nunique()))
for col in test.columns:

    print("No of unique values in column --{} is --{}".format(col, test[col].nunique()))
print("Shape of Train is {}, Shape of Test is {}".format(train.shape, test.shape))
for col in train.columns:

    print("Number of Nan in train column --{} is --{}".format(col, train[col].isna().sum()))
train['target'].value_counts().plot(kind='bar')
# to display maximum rows



pd.set_option('display.max_colwidth', -1)
train['text'].head(20)
# I am creating a feature for length of tweets



train['length'] = train['text'].str.len()



test['length'] = test['text'].str.len()
train['length']
train_1= train[train['target']==1]

train_0= train[train['target']==0]
print("Shape of train_1 is -- {}".format(train_1.shape))



print("Shape of train_0 is -- {}".format(train_0.shape))
print("Average length of text in real dataset is --{}".format(train_1["length"].mean()))



print("Average length of text in fale dataset is --{}".format(train_0["length"].mean()))
#train_1["length"].plot(kind= 'bar')
#train_0["length"].plot(kind= 'bar')
import matplotlib.pyplot as plt

from scipy.stats import norm







# No of characters of the tweet



fig,(axis0, axis1) = plt.subplots(1,2,figsize=(10,5))

axis0.hist(train_1["length"] , color='red')

#axis0.plot(train_1["length"], norm.pdf(train_1["length"],0,2))

axis0.set_title("Real Disaster tweets")





axis1.hist(train_0["length"] , color='green')

axis1.set_title("Fake Disaster tweets")



fig.suptitle('Characters in tweets')

plt.show()
# No of words in the tweet

 

fig,( axis0, axis1) = plt.subplots(1,2, figsize=(10,5))



axis0.hist(train_1["text"].str.split().map(lambda x : len(x)) , color='red')



axis1.hist(train_0['text'].str.split().map(lambda x : len(x)) , color ='green')



axis0.set_title('Real Tweets Data')



axis1.set_title('Fake Tweets Data')



fig.suptitle("Number of words in tweet")
import numpy as np

import seaborn as sns
# Average length of words in the tweet 



fig, (axis0, axis1) = plt.subplots(1,2, figsize=(10,5))



word0 = (train_1["text"].str.split().map(lambda x : len(x)))



sns.distplot(word0.map(lambda x : np.mean(x)), color= 'red', ax= axis0)

axis0.set_title('Real Tweets Data')  





word1 = train_0['text'].str.split().map(lambda x :len(x))

sns.distplot(word1.map(lambda x : np.mean(x)), color= 'green', ax = axis1)

axis1.set_title('Fake Tweets Data')  



#axis0.hist((train_1["text"].str.split().map(lambda x : len(x)).mean()))

           

#axis1.hist((train_0["text"].str.split().map(lambda x : len(x)), color = 'green'))

plt.suptitle(" Avegrage length of tweets")
# Creating corpus from the data



def create_corpus(target):

    

    corpus= []

    for values in train[train['target']== target]['text'].str.split():

        for i in values:

            corpus.append(i)

            

    return corpus
from collections import defaultdict

from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

# Creating corpus for class 0



corpus_0 = create_corpus(0)



dic=defaultdict(int)

for word in corpus_0:

    if word in stop:

        dic[word]+=1

        

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 



print(top)
x,y=zip(*top)

plt.bar(x,y)

plt.suptitle('Stop word Count')
## analyzing the tweets with target 1



corpus_1 = create_corpus(1)



dic = defaultdict(int)



for word in corpus_1:

    if word in stop:

       dic[word]+=1 

    

    top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 



print(top)
x,y = zip(*top)

plt.bar(x,y)

plt.suptitle('Stop word Count')
fig, (axis0, axis1) = plt.subplots(1,2 , figsize= (10,5))



dic= defaultdict(int)



import string

special = string.punctuation





for i in corpus_1:

    if i in special:

        dic[i]+=1

        

x,y = zip(*dic.items())

axis0.bar(x,y, color= 'red')

axis0.set_title("Characters in Real tweets")



for i in corpus_0:

    if i in special:

        dic[i] +=1

        

x,y = zip(*dic.items())

axis1.bar(x,y, color = 'green')

axis1.set_title("Characters in fake tweets")



from collections import  Counter

counter = Counter(corpus_0)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in stop) :

        x.append(word)

        y.append(count)
sns.barplot(x=y, y=x)
from sklearn.feature_extraction.text import CountVectorizer



def get_top_tweet_bigrams(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize=(10,5))

top_tweet_bigrams=get_top_tweet_bigrams(train['text'])[:10]

x,y=map(list,zip(*top_tweet_bigrams))

sns.barplot(x=y,y=x)
df = pd.concat([train,test])



print(df.shape)
import re
# Removing URLs

    

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)





    
df['text'] = df['text'].apply(lambda x : remove_URL(x))
# remove html



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

#print(remove_html(example))
df['text'] = df['text'].apply(lambda x : remove_html(x))
# remove emojis



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



df['text']=df['text'].apply(lambda x: remove_emoji(x))
# Remove Punctuations



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I am a #king"

print(remove_punct(example))
df['text']=df['text'].apply(lambda x : remove_punct(x))
!pip install pyspellchecker


!pip install pyspellchecker



import spellchecker
## Correcting spellings 



from spellchecker import SpellChecker

spell = SpellChecker()



def correct_spelling(text):

    corrected_text = []

    misspelled_word = spell.unknown(text.split())

    

    for word in text.split():

        if word in misspelled_word:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)





        
text = "corect me plese"

correct_spelling(text)
from tqdm import tqdm

from nltk.tokenize import word_tokenize

def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

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
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
MAX_LEN=50

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec
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
train1=tweet_pad[:train.shape[0]]

test=tweet_pad[train.shape[0]:]
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from nltk.corpus import stopwords

from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer

from collections import defaultdict

from collections import  Counter

plt.style.use('ggplot')

stop=set(stopwords.words('english'))

import re

from nltk.tokenize import word_tokenize

import gensim

import string

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

train['target']
X_train,X_test,y_train,y_test=train_test_split(train1,train['target'].values,test_size=0.15)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=4,epochs=15,validation_data=(X_test,y_test),verbose=2)
y_pred = model.predict(test)
y_pred.shape
y_pre=np.round(y_pred).astype(int).reshape(3263)

y_pre.shape
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sample_sub.shape
#sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})





sub = pd.DataFrame({'id': sample_sub['id'], 'target':y_pre})

sub.to_csv('submission.csv',index=False)
data = pd.read_csv("submission.csv")

data.shape
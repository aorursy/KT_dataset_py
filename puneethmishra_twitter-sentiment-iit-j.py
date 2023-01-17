import numpy as np
import random
import pandas as pd
import re
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import tensorflow as tf
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import tensorflow as tf


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
example_sent = "This is a sample sentence, showing off the stop words filtration."
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
print(stopwords.words('english'))

stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(example_sent) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 
  
print(word_tokens) 
print(filtered_sentence) 

data=pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',header=None,engine='python',encoding='latin1')
data.columns=['Sentiment','Id','Date','Query','User','Text']
train_data=data.iloc[:,-1].values
train_label=data.iloc[:,0].values
stemmer=SnowballStemmer('english')
data['Sentiment'].value_counts()
def tweet_clean(tweet):
    tweet=re.sub(r'@[A-Za-z0-9]+'," ",tweet) ##Removing the usernames
    tweet=re.sub(r'^[A-Za-z0-9.!?]+'," ",tweet) ##Removing digits and punctuations
    tweet=re.sub(r'https?://[A-Za-z0-9./]+'," ",tweet) ## removing links
    tweet=re.sub(r' +'," ",tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"\'s", " ", tweet)
    tweet = re.sub(r"\'ve", " have ", tweet)
    tweet = re.sub(r"can't", "cannot ", tweet)
    tweet = re.sub(r"n't", " not ", tweet)
    tweet = re.sub(r"\'d", " would ", tweet)
    tweet = re.sub(r"\'ll", " will ", tweet)
    tweet = re.sub(r"\'scuse", " excuse ", tweet)
    tweet = tweet.strip(' ')
    tweet = tweet.strip('. .')
    tweet = tweet.replace('.',' ')
    tweet = tweet.replace('-',' ')
    tweet = tweet.replace("’", "'").replace("′", "'").replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")
    tweet = tweet.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")
    tweet = tweet.replace("€", " euro ").replace("'ll", " will")
    tweet = tweet.replace("don't", "do not").replace("didn't", "did not").replace("im","i am").replace("it's", "it is")
    tweet = tweet.replace(",000,000", "m").replace("n't", " not").replace("what's", "what is")
    tweet = tweet.replace(",000", "k").replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")
    tweet = tweet.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")
    tweet = re.sub('\s+', ' ', tweet)
    tweet=tweet.split()
    tweet=[stemmer.stem(word) for word in tweet if word not in stop_words]
    tweet=' '.join(word for word in tweet)

    #all_stopwords = stopwords.words('english')
    return tweet.lower()

tweets_clean=[tweet_clean(tweet) for tweet in train_data]
data['Text_clean']=tweets_clean
data['No_of_Words']=[len(text.split()) for text in data['Text_clean']]

train_label[train_label==4]=1
negatives=data['Sentiment']==0
positives=data['Sentiment']==1
fig,ax =plt.subplots(nrows=1,ncols=2,figsize=(15,7.5))

sns.countplot(x=data[positives]['No_of_Words'],label='Positive',ax=ax[0])
sns.countplot(x=data[negatives]['No_of_Words'],label='Negative',ax=ax[1])
ax[0].set_title('Number of words for positive comments')
ax[1].set_title('Number of words for negative comments')
plt.tight_layout()
plt.show()
data['Words'] = data['Text_clean'].apply(lambda x:str(x).split())

top_pos = Counter([word for text in data[positives]['Words'] for word in text])
top_pos_df=pd.DataFrame(top_pos.most_common(100),columns=['Words','Counts'])

top_neg = Counter([word for text in data[negatives]['Words'] for word in text])
top_neg_df=pd.DataFrame(top_neg.most_common(100),columns=['Words','Counts'])
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(15,7.5))
sns.barplot(y='Words',x='Counts',data=top_pos_df[:20],color='deepskyblue',ax=ax[0])
sns.barplot(y='Words',x='Counts',data=top_neg_df[:20],color='coral',ax=ax[1])
ax[0].set_title("Most Frequent words in Positive tweets")
ax[1].set_title("Most Frequent words in Negative tweets")
plt.show()
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    tweets_clean, target_vocab_size=2**16
)
data_inputs = [tokenizer.encode(sentence) for sentence in tweets_clean]
Max_len=np.max([len(sentence) for sentence in data_inputs])

### Padding with 0s at the end of the sentences since 0 has no value and it wouldn't change the meaning of our sentence
data_inputs=tf.keras.preprocessing.sequence.pad_sequences(data_inputs,value=0,padding='post',maxlen=Max_len)


idx=np.random.randint(0,800000,8000)
test_idx=np.concatenate((idx,idx+800000))

X_test=data_inputs[test_idx]
y_test=train_label[test_idx]

X_test=data_inputs[test_idx]
y_test=train_label[test_idx]
X_train=np.delete(data_inputs,test_idx,axis=0)
y_train=np.delete(train_label,test_idx,axis=0)


class DCNN(tf.keras.Model):

    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=256,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name='dcnn'):
        super(DCNN,self).__init__(name=name)

        self.embeddings=layers.Embedding(vocab_size,emb_dim)

        self.bigram=layers.Conv1D(filters=nb_filters,kernel_size=2,
                                  padding='valid',activation='relu')

        self.trigram=layers.Conv1D(filters=nb_filters,kernel_size=3,
                                  padding='valid',activation='relu')

        self.fourgram=layers.Conv1D(filters=nb_filters,kernel_size=4,
                                  padding='valid',activation='relu')   

        self.pooling=layers.GlobalMaxPool1D()

        self.dense_1=layers.Dense(units=FFN_units,activation='relu')
        self.dropout=layers.Dropout(rate=dropout_rate)

        if nb_classes==2:
            self.dense_2=layers.Dense(units=1,activation='sigmoid')
        else:
            self.dense_2=layers.Dense(units=nb_classes,activation='softmax')  

    def call(self,inputs,training):
        x=self.embeddings(inputs)
        x_1=self.bigram(x)
        x_1=self.pooling(x_1)
        x_2=self.trigram(x)
        x_2=self.pooling(x_2)
        x_3=self.bigram(x)
        x_3=self.pooling(x_3)

        merged=tf.concat([x_1,x_2,x_3],axis=-1)
        merged=self.dense_1(merged)
        merged=self.dropout(merged,training)
        output=self.dense_2(merged)

        return output
VOCAB_SIZE = tokenizer.vocab_size

EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2 #len(set(train_labels))

DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NB_EPOCHS = 500
Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])


results=Dcnn.evaluate(X_test,y_test,batch_size=BATCH_SIZE)
print(results)
"""
def test_classifier(X_train, y_train, X_test, y_test, classifier):
    print("")
    print("******Classification of Twitter Sentiment Analysis******")
    classifier_name = str(type(classifier).__name__)
    print("Testing " + classifier_name)
    now = time()
    list_of_labels = sorted(list(set(y_train)))
    model = classifier.fit(X_train, y_train)
    print("Learing time {0}s".format(time() - now))
    now = time()
    predictions = model.predict(X_test)
    print("Predicting time {0}s".format(time() - now))

    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)
    print("************ Results ************")
    print("            Negative     Neutral     Positive")
    print("F1       " + str(f1))
    print("Precision" + str(precision))
    print("Recall   " + str(recall))
    print("Accuracy " + str(accuracy)) 

    return precision, recall, accuracy, f1
"""
##import random
##seed = 666
##random.seed(seed)
##from time import time
##from xgboost import XGBClassifier as XGBoostClassifier

##precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, XGBoostClassifier(seed=seed))

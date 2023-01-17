#preprocessing libraries

import pandas as pd

import numpy as np

from nltk.tokenize import word_tokenize

from tqdm import tqdm

from nltk.corpus import stopwords

#Load the dataset and combine the dataset for EDA and preprocessing
tweet= pd.read_csv('../input/nlp-getting-started/train.csv')

test=pd.read_csv('../input/nlp-getting-started/test.csv')
tweet.shape,test.shape
target = tweet["target"]

tweet.drop(["target"],axis=1,inplace=True)

#test["target"] = 0
df_train = pd.concat([tweet,test])
#convert multiline sentence in to single line sentence

import re

import string

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



def remove_punctuation(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



def remove_pattern(input_txt ):

    pattern = "@[\w]*"

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)

        

    return input_txt 



def remove_stopWord(text):

        new_text = [word.lower() for word in text.split() if((word.isalpha()==1) & (word not in stop))]

        return ' '.join(new_text)

    







stop = set(stopwords.words('english')) 

df_train["cleaned_tweet"] = df_train["text"].map(remove_URL)

df_train["cleaned_tweet"] = df_train["cleaned_tweet"].map(remove_html)

df_train["cleaned_tweet"] = df_train["cleaned_tweet"].map(remove_emoji)

df_train["cleaned_tweet"] = df_train["cleaned_tweet"].map(remove_punctuation)

df_train["cleaned_tweet"] = df_train["cleaned_tweet"].map(remove_pattern)

df_train["cleaned_tweet"] = df_train["cleaned_tweet"].map(remove_stopWord)

#Tokanized all the cleaaned tweet in our dataset

#tokenized_tweet = df_train['cleaned_tweet'].apply(lambda x: x.split())

df_train.head()
import spacy

nlp = spacy.load("en_core_web_sm")

def lemmatization(text):

    #Lemmatization:

    

    #It is a process of grouping together the inflected forms of a word so they can be analyzed as a single item, identified by the word’s lemma, or dictionary form.



     



    # Load English tokenizer, tagger, 

    # parser, NER and word vectors 

     



    # Process whole documents 

    #text = ("""My name is Shaurya Uppal. I enjoy writing articles on GeeksforGeeks checkout my other article by going to my profile section.""") 



    doc = nlp(text) 

    sent =''

    for token in doc: 

      sent +=token.lemma_+' '

    return sent



df_train["cleaned_tweet"]=df_train["cleaned_tweet"].map(lemmatization)

#lemmatization("My name is Shaurya Uppal. I enjoying writing articles on GeeksforGeeks checkout my other article by going to my profile section.")
tweet.shape[0]
train=df_train[:tweet.shape[0]]

test=df_train[tweet.shape[0]:]

train["target"]= target
from wordcloud import WordCloud

from matplotlib import pyplot as plt

normal_words =' '.join([text for text in train['cleaned_tweet'][train['target'] == 0]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()


normal_words =' '.join([text for text in train['cleaned_tweet'][train['target'] == 1]])



wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show() 
from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid=train_test_split(train.drop(["target"],axis=1),train["target"].values,test_size=0.15,random_state=23)
x_train.shape
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

# TF-IDF feature matrix

train_tf = tfidf_vectorizer.fit_transform(x_train['cleaned_tweet'])

valid_tf = tfidf_vectorizer.transform(x_valid['cleaned_tweet'])
train_tf.shape,valid_tf.shape
#Model building libraries

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from keras.initializers import Constant

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

from keras.layers import LeakyReLU,Dropout


model = Sequential()

model.add(Dense(128, kernel_initializer ='glorot_uniform',input_dim=train_tf.shape[1]))

model.add(LeakyReLU(alpha=0.01))

model.add(Dropout(0.50))

model.add(Dense(128, kernel_initializer ='glorot_uniform'))

model.add(LeakyReLU(alpha=0.01))

model.add(Dropout(0.50))

model.add(Dense(output_dim = 1, kernel_initializer ='glorot_uniform', activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adamax',metrics=['acc'])
model.summary()
from keras.callbacks import ModelCheckpoint
# checkpoint

filepath="best_weights.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

# Fit the model

model.fit(train_tf, y_train, validation_split=0.15, epochs=100, batch_size=10, callbacks=callbacks_list, verbose=1)
test_tf = tfidf_vectorizer.transform(test['cleaned_tweet'])
y_pre=model.predict(test_tf)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':test['id'].values.tolist(),'target':y_pre})

sub.to_csv('submission.csv',index=False)
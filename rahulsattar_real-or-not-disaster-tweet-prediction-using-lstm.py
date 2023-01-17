# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
import unidecode

import re

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer 

from nltk.stem import PorterStemmer 



#from keras.datasets import imdb

############ importing the dataset

df = train

############ data exploration

df['target'].value_counts().sort_values(ascending=False).plot.bar() #no class imbalance
########################### text preprocessing ####################

tweets = df['text'].to_list()



############ Defining functions for test processing ###############

def remove_accented_chars(text):

    """remove accented words like latté"""

    text = unidecode.unidecode(text)

    return text





def Punctuation(text): 

    # punctuation marks 

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''  

    # traverse the given string and if any punctuation 

    # marks occur replace it with null 

    for x in text.lower(): 

        if x in punctuations: 

            text = text.replace(x, "") 

    # Print string without punctuation 

    return(text)



#Expand Contractions

def expand_contractions(text):

    """expand shortend words like won't to would not"""

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"can't", "cannot", text)

    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)

    text = re.sub(r"\d+", "", text)

    return text



def remove_stopwords(text):

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(text)

    filtered_text = [w for w in word_tokens if len(w) >2 if w not in stop_words]

    return(filtered_text)



def lemmatize_stem(arr):

    new_arr=[]

    lemmatizer = WordNetLemmatizer() 

    ps = PorterStemmer() 

    for w in arr:

        word = lemmatizer.lemmatize(w)

        word = ps.stem(word)

        new_arr.append(word)

    return(new_arr)





######### Applying Text preprocessing ####################

def text_preprocessing(data,array,column_name):

    for i in range(len(data)):

        tweet = remove_accented_chars(data[column_name][i])

        tweet = Punctuation(tweet)

        tweet = expand_contractions(tweet)

        tweet = remove_stopwords(tweet)

        tweet = lemmatize_stem(tweet)

        array.append(tweet)
######### Applying Text preprocessing ####################

def text_preprocessing(data,array,column_name):

    for i in range(len(data)):

        tweet = remove_accented_chars(data[column_name][i])

        tweet = Punctuation(tweet)

        tweet = expand_contractions(tweet)

        tweet = remove_stopwords(tweet)

        tweet = lemmatize_stem(tweet)

        array.append(tweet)
########## Appliting preprocessing to test & train set ############

tweets = []

text_preprocessing(df,tweets,'text')



df_test = test

tweets_test = []

text_preprocessing(df_test,tweets_test,'text')
########## creating word dictonary for text2int ###################



words = [];



for list in tweets:

    tweet = list

    for word in tweet:

        if word not in words:

            words.append(word)



for list in tweets_test:

    tweet = list

    for word in tweet:

        if word not in words:

            words.append(word)



words_count ={}

for word in words:

    word_count = 0

    for l in tweets:

        if word in l:

            word_count += 1

    for l2 in tweets_test:

        if word in l2:

            word_count +=1

    words_count[word] = word_count





word2int = {}

number = 0;

for word in words:

    if words_count[word] > 2: #filtering words with less than 6 frequency        

        number +=1

        word2int[word] = number



tweets_int =[]

for list in tweets:

    tweet_int = []

    for word in list:

        if word in [*word2int]:

            tweet_int.append(word2int[word])

    tweets_int.append(tweet_int)



tweets_test_int = []           



for list in tweets_test:

    tweet_int = []

    for word in list:

        if word in [*word2int]:

            tweet_int.append(word2int[word])

    tweets_test_int.append(tweet_int)

    

######### Finding the max len of tweets ##############   

def find_max_list(list):

    list_len = [len(i) for i in list]

    print(max(list_len)) 



find_max_list(tweets_int) #Tweet length is about 20, so lets pad the sequences to 20
#####################################################################################

########### Model Building ##########################################################

#####################################################################################



########## Model Data Preparation



from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

max_tweet_length = 20



X = sequence.pad_sequences(tweets_int, maxlen=max_tweet_length)

X_train, X_test, y_train, y_test = train_test_split(X,df['target'],test_size=0.33, random_state=42)

X_predict = sequence.pad_sequences(tweets_test_int, maxlen=max_tweet_length)



######################### Model building

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers.embeddings import Embedding

from keras.layers import Dropout



top_words = len(word2int)

embedding_vecor_length = 32

model = Sequential()

model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_tweet_length))

model.add(LSTM(100))

model.add(Dropout(0.3))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2, batch_size=64)



scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))



y_hat = model.predict(X_predict, verbose=0)

y_hat_pred = []



for i in y_hat:

    if i < 0.5:

        y_hat_pred.append(0)

    else:

        y_hat_pred.append(1)  
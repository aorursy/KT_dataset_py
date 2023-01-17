import pandas as pd

import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import gensim

from gensim.models import Word2Vec

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nltk

import string

import sklearn

import string

import warnings

warnings.filterwarnings("ignore")
df_train=pd.read_excel('/kaggle/input/mh-newspred/Data_Train.xlsx')

df_test=pd.read_excel('/kaggle/input/mh-newspred/Data_Test.xlsx')



#since it is large file we can't open '.xlsx files in kaggle use the following code lines after adding the data in your kernel, It will going to display the path of your data set, So copy the followoing paths and add to load the data. 



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

   # for filename in filenames:

       # print(os.path.join(dirname, filename))

        

df_train.head()
df_test.head()
#creating a data frame and adding NEWS column  to it.

df=df_train[["STORY"]]

df["STORY"]=df["STORY"].astype(str)

df.head()



#coverting all the words to lower case for case sensitive

df["text_lower"]=df["STORY"].str.lower()

df.head()



#converting to string format and adding that as one column in 'df' data frame, 

df["text_lower"]=df["STORY"].str.lower()

df.head()





#Removing all the special characters in the data which is not required. For instance (don't,wasn't she's, , . ! etc) 

import warnings

import re

PUNCT_TO_REMOVE = string.punctuation   

def remove_punctuation(text_lower):

    return text_lower.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



df["removed_punctuations"]=df["text_lower"].apply(lambda text_lower: remove_punctuation(text_lower))

#df.head()

df["removed_punctuations"]=df["removed_punctuations"].astype(str)

text=df["removed_punctuations"]





# for removing special characters from the data ((!,@,#,$,%,^,&,*,(,) 

text = text.apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x))) 

df["special_removed"]=text.astype(str)

df.head()

#for removing numbers in the text data  

text2= df["special_removed"].apply((lambda x: re.sub(r"\d", "", x)))   #df['special_removed']=re.sub(r"\d", "", df['special_removed'])

df["numbers_removed"]=text2.astype(str)



#converting to string

df["special_removed"]=text.astype(str)

df["special_removed"]=df["special_removed"].astype(str)

#df.head()





#for removing of white spaces 

#df['numbers_removed']=df['numbers_removed'].str.strip()



# for removing extra white spaces

text3=df['numbers_removed'].apply((lambda x: re.sub(r"\s+", " ", x)))

df["whitespace_removed"]=text3.astype(str)



#removing of stop words

from nltk.corpus import stopwords

", " .join(stopwords.words('english'))



stopwords=set(stopwords.words('english'))



def remove_stopwords(sent):

    return " " .join([word for word in str(sent).split()

                      if word not in stopwords])





df["stopwords_removed"]=df["whitespace_removed"].apply(lambda sent: remove_stopwords(sent))

df.head()
import nltk

from nltk.corpus import wordnet 

from nltk.stem import WordNetLemmatizer

#df.drop(["lemmatize_words"], axis=1, inplace=True) 

lemmatizer = WordNetLemmatizer()

def lemmatize_words(sent):

    return " ".join([lemmatizer.lemmatize(word) for word in sent.split()])



df["lemmatized"] = df["stopwords_removed"].apply(lambda sent: lemmatize_words(sent))

#df.drop(["text_lemmatized"], axis=1, inplace=True) 





#removing of single characters

text4= df["lemmatized"].apply((lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x)))   #df['special_removed']=re.sub(r"\d", "", df['special_removed'])

df["singlechar_removed"]=text4.astype(str)



#removing extra spaces

text5=df['lemmatized'].apply((lambda x: re.sub(r"\s+", " ", x)))

df["space_removed"]=text5.astype(str)



#removing letters below 3 (and,how, why, etc)

text6=df['space_removed'].apply((lambda x: re.sub(r'\W*\b\w{1,3}\b', " ", x)))

df["singledouble"]=text6.astype(str)



text7=df['space_removed'].apply((lambda x: re.sub(r"\s+", " ", x)))

df["singledouble"]=text7.astype(str)



extra=df["singledouble"].str.strip()

df["strip"]=extra.astype(str)

df.head()
#Bag of words creation

from sklearn.feature_extraction.text import CountVectorizer 

bow_vectorizer = CountVectorizer(max_df=0.70, min_df=2, max_features=1000, stop_words='english')

#max_df= atleast the words needs to present in 70% of the documents

#min_df=2 (at least 2 times the words need to preset among all the documents)

#maxfeatures= set of unique words

 

# bag-of-words feature matrix

bow = bow_vectorizer.fit_transform(df['strip'])

print(bow)
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))

processed_features = vectorizer.fit_transform(df['strip'])

print(processed_features)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 

from keras.utils import to_categorical



X2=processed_features

y=df_train["SECTION"]



print(y)



X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=0)

model = LogisticRegression()

model.fit(X_train, y_train)

logpredicted_classes = model.predict(X_test)

print(confusion_matrix(y_test,logpredicted_classes))

print(classification_report(y_test,logpredicted_classes))

print(accuracy_score(y_test, logpredicted_classes))
X=bow

y=df_train["SECTION"]

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)

text_classifier.fit(X_train, y_train)



predictions = text_classifier.predict(X_test)

print(predictions)
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

print(accuracy_score(y_test, predictions))
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from gensim.models import Word2Vec

from gensim.models.keyedvectors import KeyedVectors

#loading the downloaded model

model = KeyedVectors.load_word2vec_format('/kaggle/input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin', binary=True)



#the model is loaded. It can be used to perform all of the tasks mentioned above.



# getting word vectors of a word

dog = model['dog']

dog
#perfoming similiar words for the given word from pre trained vectors

print(model.most_similar('AVENGERS'))
#perfoming similiar words for the given word from pre trained vectors

print(model.most_similar('CRICKET'))
#performing king queen magic

print(model.most_similar(positive=['woman', 'king'], negative=['man']))
#picking odd one out

print(model.doesnt_match("breakfast cereal dinner lunch".split()))
#printing similarity index

print(model.similarity('woman', 'man'))
train=df['strip']





#NOTE: #concatenating the 'strip column' because it is sereis of line,But if we want to do word embeddings all the tokens to be in one row, because it checks by word by word relation

join = ','.join(str(v) for v in train)

helo=join



#do this lines if you get get any error related objects 

#train.dropna

#train.dropna(inplace=True)

#IMPORT FROM NLTK 

from nltk.tokenize import sent_tokenize, word_tokenize

all_sentences = nltk.sent_tokenize(helo)

newsent = nltk.sent_tokenize(helo)  #This line makes all the data set of 7269 lines into one row

#print(newsent)
tokenwords = [nltk.word_tokenize(sent) for sent in newsent]   #This line makes all the data set of tokens into one row

#print(tokenwords)
classification=df_train['SECTION']

label=to_categorical([classification.values]) 

label
import gensim

from gensim.models import Word2Vec

word2vec = Word2Vec(tokenwords, min_count=2)  #min_count=2:atleast if the word repeated twice,than it will take into vector

vocabulary = word2vec.wv.vocab

#print(vocabulary)



# save model in ASCII (word2vec) format

#filename = 'embedding_word2vec.txt'

#model.wv.save_word2vec_format(filename, binary=False)
sim_words = word2vec.wv.most_similar(positive=['phone'])

sim_words

word2vec.wv.similarity('phone', 'selfies')
from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding
helo=df['strip']

from keras.preprocessing.text import Tokenizer

word_tokenizer=Tokenizer()

word_tokenizer.fit_on_texts(helo)
vocab_length = word_tokenizer.word_index

#print(vocab_length[10])
#shows the length of unique tokens and adding more space in vocab_length above 1

vocab_length = len(word_tokenizer.word_index) + 1

#print(vocab_length)
trainembedded_sentences = word_tokenizer.texts_to_sequences(helo)

#print(trainembedded_sentences)
#run each line and check

word_count = lambda sentence: len(word_tokenize(sentence))

longest_sentence = max(helo, key=word_count)

length_long_sentence = len(word_tokenize(longest_sentence))

length_long_sentence
#padded_sentences = pad_sequences(trainembedded_sentences, length_long_sentence, padding='post')

trainpadded_sentences = pad_sequences(trainembedded_sentences, maxlen=558)

#print(trainpadded_sentences)
from numpy import array

from numpy import asarray

from numpy import zeros
#here  we can load either pre-trained models or our own customized file. But make it as text file bcz to write a function for that

#here i am loading google word embeddings 

embeddings_dictionary = dict()

google_file = open('/kaggle/input/googleword2vec-as-text-file/googleVec.txt',encoding="utf8") 



for line in google_file:

    records = line.split()

    word = records[0]

    vector_dimensions = asarray(records[1:], dtype='float32')

    embeddings_dictionary [word] = vector_dimensions





google_file.close()

embedding_matrix = zeros((vocab_length, 300))

for word, index in word_tokenizer.word_index.items():

    embedding_vector = embeddings_dictionary.get(word)

    if embedding_vector is not None:

        embedding_matrix[index] = embedding_vector
#converting to matrix shape

X2=np.matrix(trainpadded_sentences)

Y2=np.matrix(label)

from sklearn.model_selection import train_test_split

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X2, Y2, test_size=0.3)

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
Model = Sequential()

embedding_layer = Embedding(vocab_length, 300, weights=[embedding_matrix], input_length=558, trainable=False)

Model.add(embedding_layer)

Model.add(Flatten())

Model.add(Dense(4, activation='softmax'))

Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print(Model.summary())
history = Model.fit(X_train, Y_train,  batch_size=32, epochs=10,  validation_data=(X_val, Y_val))
Model.evaluate(X_test, Y_test)[1]
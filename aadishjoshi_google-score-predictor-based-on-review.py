import re

import gensim

import pandas as pd

import numpy as np
data = pd.read_csv("../input/id_reviews.csv")

print(len(data))

data.head()
d = {'reviews': data["Reviews"], 'rating': data['Score']}

df = pd.DataFrame(data=d)

df.head()
def mark_sentiment(rating):

    if(rating <= 3):

        return 0

    else:

        return 1



df['sentiment'] = df['rating'].apply(mark_sentiment)

df.drop(['rating'], axis = 1, inplace=True)

df.head()
import nltk

from nltk.corpus import stopwords  #stopwords

from nltk import word_tokenize,sent_tokenize # tokenizing

from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet



# for part-of-speech tagging

from nltk import pos_tag



# for named entity recognition (NER)

from nltk import ne_chunk
def clean_reviews(review_text):

    

    review_text = re.sub("[^a-zA-Z]"," ",review_text)

    

    # 3. Converting to lower case and splitting

    word_tokens= review_text.lower().split()

    

    # 4. Remove stopwords

    le=WordNetLemmatizer()

    stop_words= set(stopwords.words("english"))     

    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words]

    

    cleaned_review=" ".join(word_tokens)

    return cleaned_review
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []

sum = 0

for review in df['reviews']:

    sents = tokenizer.tokenize(review.strip())

    sum = len(sents)

    for sent in sents:

        cleaned_sent = clean_reviews(sent)

        sentences.append(cleaned_sent.split())

print(sum)

print(len(sentences))  
max_len = 0

for m in sentences:

    if(max_len < len(m)):

        max_len = len(m)

print(max_len)
import gensim

word_2_vec_model = gensim.models.Word2Vec(sentences = sentences, size=300,window=10,min_count = 1)
word_2_vec_model.train(sentences,epochs=10,total_examples=len(sentences))
print(sentences[1:2])
vocab=word_2_vec_model.wv.vocab

vocab=list(vocab.keys())

word_vec_dict={}

for word in vocab:

  word_vec_dict[word]=word_2_vec_model.wv.get_vector(word)
import keras

from keras.preprocessing.text import one_hot,Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense , Flatten ,Embedding,Input,CuDNNLSTM,LSTM

from keras.models import Model

from keras.preprocessing.text import text_to_word_sequence
df['clean_review']=df['reviews'].apply(clean_reviews)

tok = Tokenizer()

tok.fit_on_texts(df['clean_review'])

vocab_size = len(tok.word_index) + 1

encd_rev = tok.texts_to_sequences(df['clean_review'])
pad_rev= pad_sequences(encd_rev, maxlen=max_len, padding='post')

pad_rev.shape
embed_matrix=np.zeros(shape=(vocab_size,300))

for word,i in tok.word_index.items():

  embed_vector=word_vec_dict.get(word)

  if embed_vector is not None:  # word is in the vocabulary learned by the w2v model



        embed_matrix[i]=embed_vector
print(df['sentiment'][:5])


Y=keras.utils.to_categorical(df['sentiment'])  # one hot target as required by NN.

print(Y[:5])
from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
x_train,x_test,y_train,y_test=train_test_split(pad_rev,Y,test_size=0.20,random_state=42)
from keras.initializers import Constant

from keras.layers import ReLU

from keras.layers import Dropout





model=Sequential()



model.add(Embedding(input_dim=vocab_size,output_dim=300,input_length=max_len,embeddings_initializer=Constant(embed_matrix)))

 

model.add(Flatten())

model.add(Dense(16,activation='relu'))

model.add(Dropout(0.50))

model.add(Dense(2,activation='sigmoid'))
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3),loss='binary_crossentropy',metrics=['accuracy'])
epochs=100

batch_size=64

model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))
predictions = model.predict(x_test)
(rows,cols) = predictions.shape

op = np.zeros((rows,cols))



for i in range(rows):

    for j in range(cols):

        if(predictions[i,j] < 0.5):

            op[i,j] = 0

        else:

            op[i,j] = 1

print(op[:10])
print(y_test[:10])
score, acc = model.evaluate(x_test, y_test,

                            batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
text = tok.sequences_to_texts(x_test[6:7])

print(text)
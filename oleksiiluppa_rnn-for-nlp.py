import nltk

from nltk.corpus import stopwords

import numpy as np

import pandas as pd

import re

import string

import keras.backend as K

from nltk.tokenize import word_tokenize

from sklearn import feature_extraction

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

import tensorflow.keras.regularizers as regularizers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding,LSTM,Dense,SpatialDropout1D

from tensorflow.keras.initializers import Constant

from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers import Adam

nltk.download('stopwords')

nltk.download('punkt')

stop=set(stopwords.words('english'))
test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")

tweets=pd.concat([train,test])
train.head()
pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    no_html= pattern.sub(r'',text)

    return no_html



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)





tweets['cleaned_text']=tweets['text'].apply(lambda x : remove_URL(x))

tweets['cleaned_text']=tweets['cleaned_text'].apply(lambda x : remove_html(x))

tweets['cleaned_text']=tweets['cleaned_text'].apply(lambda x : remove_punct(x))
def clean_text(text):

 

    text = re.sub('[^a-zA-Z]', ' ', text)  



    text = text.lower()  



    # split to array(default delimiter is " ") 

    text = text.split()  

    

    text = [w for w in text if not w in set(stopwords.words('english'))] 



    text = ' '.join(text)    

            

    return text



tweets['cleaned_text'] = tweets['cleaned_text'].apply(lambda x : clean_text(x))
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



tweets['cleaned_text']=tweets['cleaned_text'].apply(lambda x: remove_emoji(x))
def tokenize(data):

    tokenized=[]

    for tweet in tqdm(data["cleaned_text"]):

        words=[word for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        tokenized.append(words)

    return tokenized



tokenized=tokenize(tweets)
embedding_dict={}

with open('../input/glove6b100d/glove.6B.100d.txt','r', encoding="utf8") as f:

    for line in tqdm(f):

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
MAX_LEN=20

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(tokenized)

sequences=tokenizer_obj.texts_to_sequences(tokenized)



tweet_pad=pad_sequences(sequences, maxlen=MAX_LEN, truncating='post', padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))
num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,100))



for word,i in tqdm(word_index.items()):

    if i < num_words:

        emb_vec=embedding_dict.get(word)

        if emb_vec is not None:

            embedding_matrix[i]=emb_vec   
model=Sequential()



embedding=Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.4))

model.add(LSTM(64, dropout=0.2, recurrent_activation='tanh', recurrent_dropout=0.2))

model.add(Dense(1, activation='tanh'))





optmzr=Adam()



model.compile(loss='binary_crossentropy',optimizer=optmzr,metrics=['accuracy'])
model.summary()
train_for_model=tweet_pad[:train.shape[0]]

test=tweet_pad[train.shape[0]:]
X_train,X_test,y_train,y_test=train_test_split(train_for_model,train['target'].values,test_size=0.2)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,epochs=30,validation_data=(X_test,y_test),verbose=2)
sample_sub=pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
y_pre=model.predict(test)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})

sub.to_csv('sample_submission.csv',index=False)
sub
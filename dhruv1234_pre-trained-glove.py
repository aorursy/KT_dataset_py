import pandas as pd

train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')

train_data.head()
data = pd.concat([train_data,test_data],axis=0)

print(data.describe())

import matplotlib.pyplot as plt

plt.hist(data.target)

plt.show()
import re



def decontracted(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
import spacy

nlp = spacy.load('en')

def preprocessing(text):

  text = text.replace('#','')

  text = decontracted(text)

  text = re.sub('\S*@\S*\s?','',text)

  text = re.sub('http[s]?:(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)



  token=[]

  result=''

  text = re.sub('[^A-z]', ' ',text.lower())

  

  text = nlp(text)

  for t in text:

    if not t.is_stop and len(t)>2:  

      token.append(t.lemma_)

  result = ' '.join([i for i in token])



  return result.strip()
data.text = data.text.apply(lambda x : preprocessing(x))
from nltk.tokenize import word_tokenize

from tqdm import tqdm

from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

def create_corpus(df):

    corpus=[]

    for tweet in tqdm(df['text']):

        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]

        corpus.append(words)

    return corpus
corpus=create_corpus(data)
import numpy as np

embedding_dict={}

with open('../input/glove6b200d/glove.6B.200d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

MAX_LEN=40

tokenizer_obj=Tokenizer()

tokenizer_obj.fit_on_texts(corpus)

sequences=tokenizer_obj.texts_to_sequences(corpus)



tweet_pad=pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
word_index=tokenizer_obj.word_index

print('Number of unique words:',len(word_index))


num_words=len(word_index)+1

embedding_matrix=np.zeros((num_words,200))



for word,i in tqdm(word_index.items()):

    if i > num_words:

        continue

    

    emb_vec=embedding_dict.get(word)

    if emb_vec is not None:

        embedding_matrix[i]=emb_vec
from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense,SpatialDropout1D,Dropout

from keras.initializers import Constant

from keras.optimizers import Adam

model=Sequential()



embedding=Embedding(num_words,200,embeddings_initializer=Constant(embedding_matrix),

                   input_length=MAX_LEN,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(64,dropout=0.2, recurrent_dropout=0.2))





model.add(Dense(1, activation='sigmoid'))







optimzer=Adam(learning_rate=0.001)



model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

model.summary()
from keras.callbacks import ModelCheckpoint,EarlyStopping

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)

callbacks_list = [checkpoint,es]


tweet =data.iloc[:7613,:]

test = data.iloc[7613:,:]


train=tweet_pad[:tweet.shape[0]]

test=tweet_pad[tweet.shape[0]:]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train,tweet['target'].values,test_size=0.2)

print('Shape of train',X_train.shape)

print("Shape of Validation ",X_test.shape)
history=model.fit(X_train,y_train,batch_size=100,epochs=20,callbacks=callbacks_list,validation_data=(X_test,y_test),verbose=1)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# from keras.models import load_model

# model = load_model('../input/finalmodel/weights-improvement-21-0.84.hdf5')
sample_sub=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
y_pre = model.predict(test)

y_pre=np.round(y_pre).astype(int).reshape(3263)

sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':y_pre})

sub.to_csv('submission.csv',index=False)
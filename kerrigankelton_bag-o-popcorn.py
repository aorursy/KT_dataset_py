import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

    

print(os.listdir("../input/word2vec-nlp-tutorial"))
import zipfile



files=['/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip',

       '/kaggle/input/word2vec-nlp-tutorial/testData.tsv.zip',

       '/kaggle/input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip']



for file in files :

    zip = zipfile.ZipFile(file,'r')

    zip.extractall()

    zip.close()
train=pd.read_csv('/kaggle/working/labeledTrainData.tsv', delimiter="\t")

test=pd.read_csv('/kaggle/working/testData.tsv', delimiter="\t")
sub=pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/sampleSubmission.csv')
train.head()
print('the train data is : {} line'.format(len(train)))

print('the test data is : {} line'.format(len(test)))
train_len=train['review'].apply(len)

test_len=test['review'].apply(len)
train['word_n'] = train['review'].apply(lambda x : len(x.split(' ')))

test['word_n'] = test['review'].apply(lambda x : len(x.split(' ')))
train['length']=train['review'].apply(len)

train['length'].describe()
train['word_n'].describe()
from wordcloud import WordCloud

cloud=WordCloud(width=800, height=600).generate(" ".join(train['review'])) # join function can help merge all words into one string. " " means space can be a sep between words.

plt.figure(figsize=(15,10))

plt.imshow(cloud)

plt.axis('off')
df = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv.zip",

sep = '\t')
#Import the stopwords (common words) to be removed from the corpus



import re

import json

from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer



from nltk.stem.porter import PorterStemmer

corpus = []

s = set(stopwords.words('english'))

s.remove('not')

print("Stopwords length", len(s))
train['review']=train['review'].apply(lambda x: BeautifulSoup(x,"html5lib").get_text())

test['review']=test['review'].apply(lambda x: BeautifulSoup(x,"html5lib").get_text())
train['review']=train['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))

test['review']=test['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
train.head(3)
from wordcloud import WordCloud

cloud=WordCloud(width=800, height=600).generate(" ".join(train['review'])) # join function can help merge all words into one string. " " means space can be a sep between words.

plt.figure(figsize=(15,10))

plt.imshow(cloud)

plt.axis('off')
stops = set(stopwords.words("english"))



for i in range(0,25000) : 

    review = train.iloc[i,2] # review column : 2 

    review = review.lower().split()

    words = [r for r in review if not r in stops]

    clean_review = ' '.join(words)

    train.iloc[i,2] = clean_review
for i in range(0,25000) : 

    review = test.iloc[i,1] # review column : 1

    review = review.lower().split()

    words = [r for r in review if not r in stops]

    clean_review = ' '.join(words)

    test.iloc[i,1] = clean_review
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.preprocessing.sequence import pad_sequences

MAX_VCOCAB_SIZE = 5000

EMBEDDING_DIM = 50

MAX_SEQUENCE_LENGTH = 1500



tokenizer = Tokenizer( filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')

sequences = tokenizer.fit_on_texts(df['review'])

word_index = tokenizer.word_index

documents = tokenizer.texts_to_sequences(df['review'])

#print(word_index)

token_count = len(word_index)+1

print('Found {} unique tokens.'.format(token_count))



#print(t.word_counts)

print("Total documents ", tokenizer.document_count)

#print(t.word_index)

#print(t.word_docs)

print("max sequence length:", max(len(s) for s in documents))

print("min sequence length:", min(len(s) for s in documents))



# pad sequences so that we get a N x T matrix

data = pad_sequences(documents, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print('Shape of data tensor:', data.shape)

print(data[1])
from keras import Sequential

from keras.layers import Dense, Embedding, Flatten



model=Sequential()

model.add(Embedding(101247,65, input_length=400))

model.add(Flatten())

model.add(Dense(2,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'] )
word2vec = {}



print('Filling pre-trained embeddings...')

embedding_matrix = np.zeros((token_count, EMBEDDING_DIM))

for word, i in word_index.items():

  #if i < MAX_VOCAB_SIZE:

    embedding_vector = word2vec.get(word) #get(word) is used instead of [word] as it won't give exception in case word is not found

    if embedding_vector is not None:

      # words not found in embedding index will be all zeros.

      embedding_matrix[i,:] = embedding_vector



print("Sample embedded dimension ")

print(embedding_matrix[10][:5])
from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, GlobalAveragePooling1D 

from keras.layers import Embedding, Conv2D, GlobalMaxPooling1D 

from keras import regularizers



embedding_layer = Embedding(

  token_count,

  EMBEDDING_DIM,

  weights=[embedding_matrix],

  input_length=MAX_SEQUENCE_LENGTH,

  trainable=False)
model = Sequential()

model.add(embedding_layer)

model.add(Conv1D(filters = 64, kernel_size = 4, padding = 'same', activation='relu'))

model.add(MaxPooling1D())#kernel_size=500))

model.add(Conv1D(filters = 128, kernel_size = 3, padding = 'same',  activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.25))

model.add(MaxPooling1D())

model.add(Conv1D(filters = 256, kernel_size = 2, padding = 'same', activation='relu'))

model.add(Dropout(0.5))

model.add(MaxPooling1D())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))

model.add(Conv1D(128, 3, activation='relu'))

model.add(GlobalMaxPooling1D())



model.add(Dense(1, activation='softmax'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, df['sentiment'], test_size=0.2, random_state=42)

print(x_train.shape)
model.fit(x_train, y_train , batch_size=32, epochs=10, validation_split = 0.1)
t_loss = model.history.history['loss']

v_loss = model.history.history['val_loss']

epochs = range(1,len(t_loss)+1)

plt.plot(epochs,t_loss, 'bo', label='Training loss')

plt.plot(epochs,v_loss, 'r--', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
t_acc = model.history.history['accuracy']

v_acc = model.history.history['val_accuracy']

plt.plot(epochs,t_acc,'bo',label='Training acc')

plt.plot(epochs,v_acc,'r--', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
results = model.evaluate(x_test, y_test)

test_acc=np.round(results[1]*100,decimals=2)

#test_loss=np.round(results[2]*100,decimals=2)

print('Test accuracy is',test_acc,'%')

#print('Test loss is',test_loss,'%')
sub.to_csv('result.csv',index=False)
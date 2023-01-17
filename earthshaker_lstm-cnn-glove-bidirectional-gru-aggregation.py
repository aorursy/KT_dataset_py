import pandas as pd

from keras.models import Sequential
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Bidirectional,GRU
from keras.optimizers import Adam
from keras.utils import to_categorical
import nltk

from numpy import asarray
from numpy import zeros

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
df_train = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t")
df_test = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t")
df_sub = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv")
df_test['Sentiment'] = 999
df_combined = pd.concat([df_train, df_test])
df_combined.head()
def CleanText(sentence):
    lemmaList = []
  
    tokens = sentence.split()
  
    for token in tokens:
        lemmaToken = wordnet_lemmatizer.lemmatize(token)
        lemmaToken = lemmaToken.lower()
        lemmaList.append(lemmaToken)
    
    lemmaSentence = " ".join(lemmaList)  
  
    return lemmaSentence
df_combined['Cleaned'] = df_combined['Phrase'].apply(lambda x : CleanText(x))
df_combined.head()
df_test = df_combined[df_combined['Sentiment'] == 999]
df_train = df_combined[df_combined['Sentiment'] != 999]

y_train = df_train['Sentiment']
y_test = df_test['Sentiment']

y_train=to_categorical(y_train)

df_train = df_train.drop(['Sentiment'],axis=1)
df_test = df_test.drop(['Sentiment'],axis=1)
# Split Train set to Train-Validation set
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val=train_test_split(df_train,y_train,test_size=0.3,random_state=123, stratify=y_train)

train_sentences = X_train['Cleaned']
validation_sentences = X_val['Cleaned']
test_sentences = df_test['Cleaned']
# Data feature extraction on X_train

# Get total number of unique words
collection = ' '.join(map(str, train_sentences))

MAXCHARS = len(set(collection.split()))

# Get maximum length of sentence to pad the rest to
lengths = []

for sentence in train_sentences:
    length = len(sentence)
    lengths.append(length)
    
MAXLEN = max(sorted(lengths))
# Creating input to LSTM model

tokenizer = Tokenizer(num_words=MAXCHARS)
tokenizer.fit_on_texts(train_sentences)

X_train = tokenizer.texts_to_sequences(train_sentences)
X_val = tokenizer.texts_to_sequences(validation_sentences)
X_test = tokenizer.texts_to_sequences(test_sentences)

# padding to create uniform length

X_train = sequence.pad_sequences(X_train, maxlen=MAXLEN, padding='post')
X_val = sequence.pad_sequences(X_val, maxlen=MAXLEN, padding='post')
X_test = sequence.pad_sequences(X_test, maxlen=MAXLEN, padding='post')
# LSTM parameters

batch_size = 256
timestep = MAXLEN
features = MAXCHARS
num_classes = 5
model=Sequential()
model.add(Embedding(MAXCHARS,50,mask_zero=True))
model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(64,dropout=0.2, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=3, batch_size=batch_size, verbose=1)
y_test = model.predict_classes(X_test,verbose=1)
model2 = Sequential()
model2.add(Embedding(MAXCHARS,100,input_length=MAXLEN))

model2.add(Conv1D(64, kernel_size=3, padding='same', activation='relu', strides=1))
model2.add(GlobalMaxPooling1D())

model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.2))

model2.add(Dense(num_classes,activation='softmax'))

model2.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model2.summary()
model2.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=3, batch_size=256, verbose=1)
y_test2 = model2.predict_classes(X_test,verbose=1)
# Reading in Glove and getting embeddings for the words

embeddings_index = {}

# Reading in Glove
with open('../input/glove-vectors/glove.6B.100d.txt','r', encoding="utf8") as f:
    lines = f.readlines()
    for line in lines:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Creating embedding matrix for our vocab
embedding_matrix = zeros((MAXCHARS, 100))    

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
model3 = Sequential()
model3.add(Embedding(MAXCHARS,100,input_length=MAXLEN, weights=[embedding_matrix], trainable=False))
model3.add(Dropout(0.2))
model3.add(Bidirectional(GRU(128,return_sequences=True)))
model3.add(Bidirectional(GRU(64,return_sequences=False)))
model3.add(Dropout(0.2))

model3.add(Dense(num_classes,activation='softmax'))

model3.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

model3.summary()
model3.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=5, batch_size=512, verbose=1)
y_test3 = model3.predict_classes(X_test,verbose=1)
sub_all=pd.DataFrame({'model1':y_test,'model2':y_test2,'model3':y_test3})
pred_mode=sub_all.agg('mode',axis=1)[0].values

pred_mode=[int(i) for i in pred_mode]

df_sub['Sentiment']=pred_mode
df_sub.to_csv('mySubmission.csv',index=False)
df_sub.head()

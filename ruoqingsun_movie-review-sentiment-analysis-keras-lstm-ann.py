# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 

train = pd.read_csv('../input/train.tsv',sep = '\t')
test = pd.read_csv('../input/test.tsv', sep = '\t')
print("Train set: {0}".format(train.shape))
print("Test set: {0}".format(test.shape))

df = pd.concat([train, test])
print("All df set: {0}".format(df.shape))

df.head()
sub = pd.read_csv('../input/sampleSubmission.csv', sep = ',')
print("Submission: {0}".format(sub.shape))

sub.head()
x = train.groupby(['Sentiment'])['PhraseId'].count()
x.plot.bar()
print("Training set distribution: ", train.groupby(['Sentiment']).size()/train.shape[0])
import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    review_lemma=[]
    for word in text.split():
        word_lemma = wordnet_lemmatizer.lemmatize(word)
        review_lemma.append(word_lemma)
    review_lemma=' '.join(review_lemma)
    return review_lemma
train['clean_phrase'] = train['Phrase'].apply(clean_text)
test['clean_phrase'] = test['Phrase'].apply(clean_text)
df['clean_phrase'] = df['Phrase'].apply(clean_text)
train.head()
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk import FreqDist
train_text=train.clean_phrase.values
test_text=test.clean_phrase.values
target=train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)
X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)
all_words = ' '.join(X_train_text)
word2count = {}
for word in all_words.split():
    if word not in word2count:
        word2count[word] = 1
    else:
        word2count[word] += 1
print("Number of unique words: ", len(word2count.keys()))
df['length_review'] = df['clean_phrase'].apply(lambda x: len(x.split()))
print("Max phrase length: ", max(df['length_review']))
d = pd.DataFrame(list(word2count.items()), columns=['word', 'count'])
d.head()
all_phrases = [X_train_text]
all_phrases
# from sklearn.feature_extraction.text import TfidfTransformer

# sklearn_tfidf = TfidfTransformer()
# sklearn_representation = sklearn_tfidf.fit_transform(all_phrases)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
MAX_REVIEW_LENGTH = 49
FEATURE_LENGTH = 12011
BATCH_SIZE = 1000
EPOCHS = 100
NUM_CLASSES = 5
tokenizer = Tokenizer(num_words = FEATURE_LENGTH)
tokenizer.fit_on_texts(list(np.concatenate((train_text, test_text), axis=0)))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)
X_train = pad_sequences(X_train, maxlen=MAX_REVIEW_LENGTH)
X_val = pad_sequences(X_val, maxlen=MAX_REVIEW_LENGTH)
X_test= pad_sequences(X_test, maxlen=MAX_REVIEW_LENGTH)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
model=Sequential()
model.add(Embedding(FEATURE_LENGTH,250,mask_zero=True))
model.add(LSTM(128,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(NUM_CLASSES,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
y_pred = model.predict_classes(X_test)
test["Sentiment"] = y_pred
test[['PhraseId', 'Sentiment']].to_csv('submission_lstm.csv', index = False)
from keras.layers import Flatten
ann_model = Sequential()
ann_model.add(Embedding(FEATURE_LENGTH,250, input_length=MAX_REVIEW_LENGTH))
ann_model.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
ann_model.add(Flatten())
ann_model.add(Dense(output_dim = 50, activation='tanh'))
ann_model.add(Dense(output_dim = 10, activation = 'relu'))
ann_model.add(Dense(NUM_CLASSES,activation='softmax'))
ann_model.compile(optimizer=Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
ann_model.summary()
ann_history = ann_model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size = BATCH_SIZE, epochs = EPOCHS)
y_pred = ann_model.predict_classes(X_test)
test["Sentiment"] = y_pred
test[['PhraseId', 'Sentiment']].to_csv('submission_ann.csv', index = False)

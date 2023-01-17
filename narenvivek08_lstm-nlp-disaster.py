import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
train.head()
test.head()
print("Total training samples:",len(train['id']))
print("Total testing samples:", len(test['id']))
print("Missing keywords in training:",train['keyword'].isnull().sum())
print("Missing keywords in testing:", test['keyword'].isnull().sum())
missing_col=['keyword','location']
ax=sns.barplot(x=missing_col, y=train[missing_col].isnull().sum())
ax.set_title('Training set')
ax.set_ylabel('Missing values')

plt.show()
ax=sns.barplot(x=missing_col, y=test[missing_col].isnull().sum())
ax.set_title('Test set')
ax.set_ylabel('Missing values')
plt.show
sns.countplot(y=train['target'])
sns.countplot(y=train['location'], order=train['location'].value_counts().iloc[:10].index)
plt.title('Top 10 countries')
plt.show()

sns.countplot(y=train['keyword'], order=train['keyword'].value_counts().iloc[:20].index)
plt.title('Top 20 keywords in training set')
plt.show
kw=train[train['target']==1]['keyword'].value_counts().head(10)
kw1=train[train['target']==0]['keyword'].value_counts().head(10)
plt.figure(figsize=(22,8))
plt.subplot(121)
sns.barplot(kw, kw.index, color='b')
plt.title('Real keywords')
plt.subplot(122)
sns.barplot(kw1, kw1.index, color='r')
plt.title('Fake keywords')
plt.show
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
train['clean_text']='NaN'
train.head()
i=0
for sentence in train['text']:
    train['clean_text'][i]=clean_text(sentence)
    i=i+1
    
train.head()
test['clean_text']='NaN'
i=0
for sentence in test['text']:
    test['clean_text'][i]=clean_text(sentence)
    i=i+1
    
test.head()
number=0
for sentence in train['clean_text']:
    if len(sentence)>number:
        number=len(sentence)
    else:
        continue
print(number)
    
train['keyword'].fillna('Unknown', inplace = True) 
train.head()
test['keyword'].fillna('Unknown',inplace=True)
test.head()
train['final_text']='NaN'
train['final_text']=train['clean_text']+train['keyword']
train.head(10)
test['final_text']='NaN'
test['final_text']=test['clean_text']+test['keyword']
test.head()
data=train['final_text'].values
valid_data=test['final_text'].values
target=train['target'].values
X_train, X_test, Y_train, Y_test= train_test_split(data, target,test_size=0.1, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
tokenizer=Tokenizer(num_words=3000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index
sequences=tokenizer.texts_to_sequences(X_train)
padded_train=pad_sequences(sequences,160,truncating='post')
print(padded_train.shape)
sequences_test=tokenizer.texts_to_sequences(X_test)
padded_test=pad_sequences(sequences_test,160,truncating='post')
model= tf.keras.Sequential([
    tf.keras.layers.Embedding(3000,32,input_length=160),
    tf.keras.layers.SpatialDropout1D(0.4),
    
    tf.keras.layers.LSTM(64, return_sequences=True),#added for 3rd sub
    tf.keras.layers.LSTM(32),
    
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
opt=keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
history=model.fit(padded_train, Y_train, epochs=10, validation_data=(padded_test, Y_test),batch_size=32)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
sequences_valid=tokenizer.texts_to_sequences(valid_data)
padded_valid=pad_sequences(sequences_valid,160,truncating='post')
results=model.predict(padded_valid)
results=results.round()
results = np.array(results, dtype='int')
sub=pd.DataFrame()
sub['id']=test['id']
sub.head()
sub['target']=results
sub.head()
sub.to_csv('submission4.csv', index=False)
from collections import Counter
Counter(sub['target'])

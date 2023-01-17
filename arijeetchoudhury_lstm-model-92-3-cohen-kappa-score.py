import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#load the data
data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
data.head()
#get rid of useless columns
data = data[['v1','v2']]
data.head()
data.v1.value_counts()
sns.countplot(data.v1)
def set_label(value):
    return 0 if value == 'ham' else 1
data['v1'] = data['v1'].apply(set_label)
data.head()
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator)
def remove_tags(text):
  return re.sub('<.*?>'," ",text)
def remove_numbers(text):
  return re.sub('[0-9]+','number',text)
def remove_urls(text):
  return re.sub('https?:\S+','httpaddr',text)

def remove_emails(text):
    return re.sub('\S+@\S+','email',text)
data['v2'] = data['v2'].apply(remove_urls)
data['v2'] = data['v2'].apply(remove_tags)
data['v2'] = data['v2'].apply(remove_emails)
data['v2'] = data['v2'].apply(remove_punctuation)
data['v2'] = data['v2'].apply(remove_numbers)
data['v2'] = data['v2'].apply(lambda word : word.lower())
data.head()
from nltk.corpus import stopwords
stops = stopwords.words('english')
def remove_stopwords(text):
    cleaned = []
    for word in text.split():
        if word not in stops:
            cleaned.append(word)
    return " ".join(cleaned)
data['v2'] = data['v2'].apply(remove_stopwords)
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
  lemmas = []
  for word in text.split():
    lemmas.append(lemmatizer.lemmatize(word))
  return " ".join(lemmas)
data['v2'] = data['v2'].apply(lemmatize_words)
data = data.sample(frac=1).reset_index(drop=True)
X = data['v2'].values
y = data['v1'].values
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(X)
word_to_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X)
vocab_size = len(word_to_index)
max_length = 50
embedding_dim = 100
padded_sequences = pad_sequences(sequences,maxlen=max_length,padding='post',truncating='post')
embeddings_index = {};
with open('glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;
print(embeddings_matrix.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,SpatialDropout1D,LSTM,Bidirectional,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
model = Sequential([
    Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128,recurrent_dropout=0.2,dropout=0.2)),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])
optimizer = Adam(learning_rate=0.01)
callbacks = ReduceLROnPlateau(monitor='val_accuracy',patience=2,factor=0.5,min_lr=0.00001)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.summary()
#split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_sequences,y,test_size=0.3,random_state=1)
print('No. of training samples:',len(X_train))
print('No. of testing samples:',len(X_test))
epochs = 10
history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=64,callbacks=[callbacks])
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
train_stats = model.evaluate(X_train,y_train)
test_stats = model.evaluate(X_test,y_test)
print('training accuracy:',train_stats[1]*100)
print('testing accuracy:',test_stats[1]*100)

y_pred = model.predict_classes(X_test)
print(classification_report(y_test,y_pred))
print('Confusion matix:\n',confusion_matrix(y_test,y_pred))
print('Cohen-kappa score:',cohen_kappa_score(y_test,y_pred))

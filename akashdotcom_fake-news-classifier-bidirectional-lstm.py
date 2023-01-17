import pandas as pd
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

df = pd.read_csv('../input/fake-news/train.csv')
df.head()
df = df.dropna()
X = df.drop('label', axis = 1)
y = df['label']
y.value_counts()
X.shape
y.shape
#vocabulary size
voc_size = 5000
messages = X.copy()
messages['title'][0]
messages.reset_index(inplace = True)
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    #print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
corpus[0:5] #stemming and removed stopwords
onehot_repr = [one_hot(words, voc_size) for words in corpus]
onehot_repr
#we are assigning input length to be 30
sent_length = 30
embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length)
print(embedded_docs)
#for first news text
embedded_docs[0]
#giving vector features

embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length = sent_length))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.4))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()
import numpy as np
X_final = np.array(embedded_docs)
y_final = np.array(y)
X_final.shape, y_final.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size = 0.2, random_state = 42)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 25, batch_size = 64)
y_pred = model.predict_classes(X_test)
y_pred
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

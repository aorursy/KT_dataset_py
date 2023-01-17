# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv")
test = pd.read_csv("/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv")
train.head()
train.nunique()
train["No_of_topics"] = train["Computer Science"]+train["Physics"]+train["Mathematics"]+train["Statistics"]+train["Quantitative Biology"]+train["Quantitative Finance"]
train[train["No_of_topics"] > 1]   
train.No_of_topics.value_counts()
train["content"] = train["TITLE"]+train["ABSTRACT"]
train.drop(labels = ["ID","TITLE","ABSTRACT","No_of_topics"],axis=1,inplace = True)
train.head()
from collections import Counter
def vocab(texts):
    cnt = Counter()
    for row in texts.values:
        for i in row.split():
            cnt[i] += 1
    return len(cnt)
vocab_size = vocab(train.content)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics


labels = ['Computer Science', 'Physics', 'Mathematics','Statistics', 
          'Quantitative Biology', 'Quantitative Finance']

for label in labels:
    print(label)
    print('')
    print('Value counts:')
    print(train[label].value_counts())

    X = train['content']
    y = train[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)
    
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC()),
    ])

    text_clf.fit(X_train, y_train)  

    predictions = text_clf.predict(X_test)

    print(metrics.confusion_matrix(y_test,predictions))
    print('')
    print(metrics.classification_report(y_test,predictions))
    print('')
    print('')
    print('')
    print('')
import matplotlib.pyplot as plt

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
train_labels = train[['Computer Science', 'Physics', 'Mathematics','Statistics','Quantitative Biology', 'Quantitative Finance']]
train_labels.sum(axis=0).plot.bar()
import re
def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

X = []
sentences = list(train["content"])
for sen in sentences:
    X.append(preprocess_text(sen))
y = train_labels.values
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 200

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_file = open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 200))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
from keras.layers import Embedding,Dense,GlobalMaxPool1D,Dropout,Flatten,Bidirectional,LSTM
from keras.models import Sequential
# Model 1
# deep_inputs = Input(shape=(maxlen,))
# embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], trainable=False)(deep_inputs)
# LSTM_Layer_1 = LSTM(128)(embedding_layer)
# maxpool = GlobalMaxPooling1D()
# dense_layer2 =  Dense(128, activation='relu')(maxpool)
# dense_layer_1 = Dense(6, activation='sigmoid')(LSTM_Layer_1)
# model = Model(inputs=deep_inputs, outputs=dense_layer_1)
# Model 2
model=Sequential([Embedding(vocab_size,200,input_length=maxlen,weights=[embedding_matrix], trainable=False),
                 Bidirectional(LSTM(128,return_sequences=True)),
                 GlobalMaxPool1D(),
                  Dense(128,activation = 'relu'),
                 Dense(64,activation='relu'),
                  Dense(6,activation='sigmoid')
                 ])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)
history = model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
test.head()
test['content'] = test["TITLE"]+test["ABSTRACT"]
test.drop(labels = ["ID","TITLE","ABSTRACT"],axis=1,inplace = True)
test.head()

test_df = []
rows = list(test.content)
for sent in rows:
    test_df.append(preprocess_text(sent))
    
#     return rows
# rows
# from keras.preprocessing.text import pad_sequences,texts_to_sequences
tokenizer.fit_on_sequences(test_df)
X_test = tokenizer.texts_to_sequences(test_df)
X_test = pad_sequences(X_test,maxlen = 200,padding = 'post')

preds = model.predict(X_test)
for arr in preds:
    for i in range(len(arr)):
        if arr[i]>0.5:
            arr[i] = 1
        else:
            arr[i] = 0

preds = preds.astype("int32")
preds
df = pd.DataFrame(data = preds,columns = ['Computer Science', 'Physics', 'Mathematics','Statistics','Quantitative Biology', 'Quantitative Finance'])
df.head()
sample = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv")
sample
final_df = pd.DataFrame({"ID":sample.ID,})
final = pd.concat([final_df,df],axis=1)
final.to_csv("submission.csv",index = False)
print(final.shape)
final.head()


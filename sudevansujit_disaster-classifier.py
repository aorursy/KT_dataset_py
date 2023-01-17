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
import pandas as pd
import numpy as np
data = pd.read_csv("../input/nlp-getting-started/train.csv")
data.head()
# Checking Shape
data.shape
data['keyword'].value_counts()
data['target'].value_counts()
import re
count = 0
for i in data['text']:
    res = re.findall(r'#[\w]+', i)
    print(res)
    count += 1
    if count == 10:
        break
data.head(10)
data = data.drop(['id','keyword', 'location'],axis=1)
data.shape
import string
def clean_text(text):
    text = text.lower()
    for i in string.punctuation:
        if i in text:
            text = text.replace(i, '')
    return text

    
clean_text("Hello sam ; I wanna'a talk to u")
data['text'] = data['text'].apply(clean_text)
data.head()
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)
def remove_stop(text):
    text = [word for word in text.split() if word not in stop_words ]
    text = ' '.join(text)
    return text
remove_stop('Hello world i can see you')
data['text'] = data['text'].apply(remove_stop)
data.head()
X = data['text']
y = data['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.8)
X_train.shape, X_test.shape
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range = (1,2), max_features = 5000)
tfidf.fit(X_train)
X_train_1 = tfidf.transform(X_train)
X_test_1 = tfidf.transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model):
    
    # Classification Report
    y_pred = model.predict(X_test_1)
    print(classification_report(y_pred, y_test))
    print('*'*50)
    
    #Accuracy Score
    print("Accuracy Score :", accuracy_score(y_pred, y_test))
    print("*"*50)
    
    # Confusion Matrix Heatmap
    cnf_matrix = confusion_matrix(y_pred, y_test)
    labels = ['Negative', 'Positive']
    plt.figure(figsize = (8,8))
    sns.heatmap(cnf_matrix, 
              annot = True, 
              cmap = "Blues", 
              fmt = '',
              xticklabels = labels,
              yticklabels = labels)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.show()
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
X_train_1.shape
y_train.shape
# BernoullNB
bnb_model = BernoulliNB(alpha = 2)
bnb_model.fit(X_train_1, y_train)

evaluate_model(bnb_model)
# Multinomial NB
mnb_model = MultinomialNB(alpha = 2, fit_prior = False)
mnb_model.fit(X_train_1, y_train)

evaluate_model(mnb_model)
# Linear SVC
SVC_model = LinearSVC()
SVC_model.fit(X_train_1, y_train)

evaluate_model(SVC_model)
# Logistic Regression
LR_model = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LR_model.fit(X_train_1, y_train)

evaluate_model(LR_model)
labels = list(data['target'].values)
texts = list(data['text'].values)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
maxlen = 100
training_samples = 200
validation_samples = 100
max_words = 10000

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

print("Unique Words", len(word_index))
data = pad_sequences(sequences, maxlen = maxlen)
labels = np.asarray(labels)
print("Shape of data tensor ", data.shape)
print("Shape of label tensor ", labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]
X_train = data[:training_samples]
y_train = labels[:training_samples]

X_val = data[training_samples:training_samples+validation_samples]
y_val = labels[training_samples:training_samples+validation_samples]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences = True)) # Weights will remain the same
model.add(SimpleRNN(32))
model.summary()
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(X_train, y_train, epochs = 100, batch_size = 128, validation_split = 0.2)
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label = 'Training Acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation Acc')
plt.title("Training and Validation Accuracy")
plt.show()

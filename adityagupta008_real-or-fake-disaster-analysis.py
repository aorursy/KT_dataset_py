#Importing libraries
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Reading the data
data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
data.head()
# Dropping the columns which will have no impact in the sentiment analysis
train = data.drop(['keyword', 'location', 'id'], axis= 1)
train.head()
# Looking at our whole training data shape
train.shape
#Observe the values and notice that there are use of urls. Also, as these are tweets, so # and @ must be used frequently. Let us remove those
np.array(train['text'].values)
#Removing urls
import re
def remove_urls(dataframe):
    url_pattern = r'https*?:\/\/.*[\r\n]*'
    texts = dataframe['text'].values
    for i in range(len(texts)):
        texts[i] = re.sub(url_pattern, repl= '', string= texts[i]).strip()
    dataframe['text'] = texts
remove_urls(train)
# Now the urls are removed
np.array(train['text'].values)
# removing # and @ symbols
def remove_hashtags_and_at_the_rate(dataframe):
    url_pattern = r'[#@]+'
    texts = dataframe['text'].values
    for i in range(len(texts)):
        texts[i] = re.sub(url_pattern, repl= '', string= texts[i]).strip()
    dataframe['text'] = texts
remove_hashtags_and_at_the_rate(train)
np.array(train['text'].values)
X = np.array(train['text'].values)
y = np.array([{ 'cats': { '1': target == 1, '0': target == 0}} for target in train['target']])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle= True, random_state= 42)
print('X_train shape : ', X_train.shape)
print('X_test shape : ', X_test.shape)
print('y_train shape : ', y_train.shape)
print('y_test shape : ', y_test.shape)
# Initialising a blank model
import spacy
    
nlp = spacy.blank('en')

textcat = nlp.create_pipe('textcat', config= {'exclusive_classes': True,
                                              'architechture':'bow'})

nlp.add_pipe(textcat)
textcat.add_label('1')
textcat.add_label('0')
from spacy.util import minibatch
import random

def train_model(model, training_data):
    optimizer = nlp.begin_training()
    
    losses= {}
    random.shuffle(training_data)
    batches = minibatch(training_data, size= 8)

    for batch in batches:
        texts, labels = zip(*batch)
        nlp.update(texts, labels, sgd= optimizer, losses= losses)
        
    return losses['textcat']
def predict(model, texts): 
    docs = [model.tokenizer(text) for text in texts]
    textcat = model.get_pipe('textcat')
    scores, _ = textcat.predict(docs)
    predicted_class = scores.argmax(axis= 1)
    
    return predicted_class
from sklearn.metrics import f1_score

def evaluate(model, texts, labels):
    
    predicted_class = predict(model, texts)
    true_class = [int(label['cats']['1']) for label in labels]
    correct_predictions = true_class == predicted_class
    
    accuracy = correct_predictions.mean()
    f1 = f1_score(true_class, correct_predictions.astype('int'))
    
    return accuracy, f1
n_epochs = 5

training_data = list(zip(X_train, y_train))

for epoch in range(n_epochs):
    loss = train_model(nlp, training_data)
    accuracy, f1 = evaluate(nlp, X_test, y_test)
    print(f'Epoch {epoch+1} Loss : {loss:.3f} Accuracy : {accuracy:.3f}, F1-Score : {f1:.3f}')
import spacy

nlp = spacy.load('en_core_web_lg')

train_vectors = np.array([nlp(text).vector for text in train['text']])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_vectors, train['target'], test_size= 0.2, shuffle= True, random_state= 42)
from sklearn.svm import LinearSVC

svc = LinearSVC(dual= False, max_iter= 10000, random_state= 1)
svc.fit(X_train, y_train)

print(f'Accuracy Score : {svc.score(X_test, y_test):.3f}')
from sklearn.metrics import f1_score

f1_score(y_test, svc.predict(X_test))
from xgboost import XGBClassifier

xgb = XGBClassifier( n_jobs= -1)
xgb.fit(X_train, y_train)
f1_score(y_test, xgb.predict(X_test))
from sklearn.svm import SVC
m = SVC()
m.fit(X_train, y_train)
f1_score(y_test, m.predict(X_test))
# final_model1 = SVC()
# final_model1.fit(train_vectors, train['target'])
# test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
# ids = test.id.values
# test = test.drop(['keyword', 'location', 'id'], axis= 1)
# remove_urls(test)
# remove_hashtags_and_at_the_rate(test)
# test.head()
# test_vectors = np.array([nlp(text).vector for text in test['text']])
# preds = final_model1.predict(test_vectors)
# submission = pd.DataFrame(columns= ['id', 'target'], data = zip(ids, preds))
# submission.head()
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np
tokenizer = Tokenizer()
corpus = train['text'].values
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index)+1
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    input_sequences.append(token_list)
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen= max_seq_len, padding= 'pre'))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_sequences, train['target'].values, test_size= 0.2, shuffle= True, random_state= 42)
from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
model = Sequential()
model.add(Embedding(total_words, 100, input_length= max_seq_len))
model.add(Bidirectional(LSTM(200, return_sequences= True)))
model.add(Dropout(0.4))
model.add(Bidirectional(LSTM(150)))
model.add(Dropout(0.5))
model.add(Dense(total_words/2, activation= 'relu', kernel_regularizer= regularizers.l2(0.01)))
model.add(Dense(total_words, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(loss= 'binary_crossentropy', optimizer= 'adam', metrics= ['accuracy', f1_m])
model.fit(X_train, y_train, epochs= 15, validation_data= (X_test, y_test))
from sklearn.metrics import f1_score
p = model.predict(X_test)
p = np.squeeze(p)
# def limit(x):
#     return 1 if x > 0.4 else 0
# p = list(map(limit, p))
f1_m(y_test.astype('float32'), p.astype('float32'))




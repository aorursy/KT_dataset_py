import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import tensorflow as tf

from tqdm import tqdm

tqdm.pandas()

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding,GRU,Bidirectional,Convolution1D,Dense,GlobalMaxPool1D,Dropout,Bidirectional,LSTM

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from gensim.models import KeyedVectors

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/basic-clean-complete-data/Consumer Clean Data.csv',encoding = 'latin-1')

df.head()
df['Company response to consumer'].value_counts()
df1 = df[df['Company response to consumer']=='Closed with explanation']

df2 = df[df['Company response to consumer']=='Closed with non-monetary relief']

df3 = df[df['Company response to consumer']=='Closed with monetary relief']



df1 = df1[:15000]

df2 = df2[:15000]



df = pd.concat([df1,df2,df3])
df['Company response to consumer'].value_counts()
def renaming(text):

    if text == 'Closed with explanation':

        return 'Closed with non-monetary relief'

    else:

        return text

    

df['Company response to consumer'] = df['Company response to consumer'].progress_apply(renaming)
df['Company response to consumer'].value_counts()
df.drop_duplicates(subset='Consumer complaint narrative',inplace = True)
df['Company response to consumer'].value_counts()
df.shape
df.head()
#df = df[:130000]

df = df.sample(frac=1).reset_index(drop=True)

df.head()
x = df['Embeddings Clean'].astype(str)

y = df['Company response to consumer']



le = LabelEncoder()



y = le.fit_transform(y)

#y = to_categorical(y)
le.classes_
tokenizer = Tokenizer()



tokenizer.fit_on_texts(x)



seq = tokenizer.texts_to_sequences(x)



pad_seq = pad_sequences(seq,maxlen = 300,padding='post',truncating='post')
vocab_size = len(tokenizer.word_index)+1

vocab_size
word2vec = KeyedVectors.load_word2vec_format('/kaggle/input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin', \

        binary=True)

print('Found %s word vectors of word2vec' % len(word2vec.vocab))
embedding_matrix = np.zeros((vocab_size, 300))

for word, i in tokenizer.word_index.items():

    if word in word2vec.vocab:

        embedding_matrix[i] = word2vec.word_vec(word)
import gc

del df

del x

del seq

gc.collect()
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
model = Sequential()

model.add(Embedding(vocab_size,300,input_length=300,weights = [embedding_matrix],trainable = False))

model.add(Bidirectional(LSTM(50,return_sequences=True)))

#model.add(Bidirectional(LSTM(32,return_sequences = True)))

model.add(Convolution1D(64,3,activation = 'relu'))

#model.add(Convolution1D(128,3,activation = 'relu'))

model.add(GlobalMaxPool1D())

#model.add(Dense(128,activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(1,activation = 'sigmoid'))



model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy',precision_m,recall_m,f1_m])
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(pad_seq,y,test_size = 0.1,random_state= 42)
model.fit(x_train,y_train,batch_size = 32,epochs = 5,validation_split = 0.1)
predictions = model.predict(x_test)
from sklearn.metrics import f1_score,precision_score,recall_score
recall = []

for thresh in np.arange(0,0.9,0.01):

    thresh = np.round(thresh,2)

    print('Recall Score at threshold {0} is {1}'.format(thresh,recall_score(y_test,(predictions>thresh).astype(int))))

    recall.append(recall_score(y_test,(predictions>thresh).astype(int)))
precision = []

for thresh in np.arange(0,0.9,0.01):

    thresh = np.round(thresh,2)

    print('Precision Score at threshold {0} is {1}'.format(thresh,precision_score(y_test,(predictions>thresh).astype(int))))

    precision.append(precision_score(y_test,(predictions>thresh).astype(int)))
f1 = []

for thresh in np.arange(0,0.9,0.01):

    thresh = np.round(thresh,2)

    print('F1 Score at threshold {0} is {1}'.format(thresh,f1_score(y_test,(predictions>thresh).astype(int))))

    f1.append(f1_score(y_test,(predictions>thresh).astype(int)))
x_axis = range(len(f1))

plt.plot(x_axis,recall,label = 'Recall')

plt.plot(x_axis,precision,label = 'Precision')

plt.plot(x_axis,f1,label = 'F1 Score')

plt.legend()



plt.show()
x_test_prediction = []

for i in tqdm(range(len(predictions))):

    if predictions[i][0]>0.5:

        x_test_prediction.append(int(1))

    else:

        x_test_prediction.append(int(0))
print(classification_report(y_test,x_test_prediction))
print(accuracy_score(y_test,x_test_prediction))
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.heatmap(confusion_matrix(y_test,x_test_prediction),annot = True,fmt = 'g')
df = pd.read_csv('/kaggle/input/testapril/complaints-2020-05-07_09_29.csv')

df.head()
df.shape
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

import re
def removal(text):

    text = re.sub('[^A-Za-z]',' ',text)

    text = re.sub('xxxx','',text)

    text = re.sub('xxx','',text)    

    text = re.sub('xx','',text)

    text = re.sub('xx\/xx\/\d+','',text)

    #text = re.sub('UNKNOWN   UNKNOWN','UNKNOWN',text)

    text = re.sub('\n',' ',text)

    text = re.sub(' +',' ',text)

    

    return text



stop_words = stopwords.words('english')

words = []

for i in tqdm(range(len(stop_words))):

    words.append(re.sub('[^A-Za-z]','',stop_words[i]))

    

stop_words = list(set(stop_words+words))



lem = WordNetLemmatizer()

pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')





def cleaning(text):

    text = text.lower()

    text = pattern.sub(' ', text)

    text = removal(text)

    text = word_tokenize(text)

    text = [lem.lemmatize(w,'v') for w in text]

    text = ' '.join(text)

    text = re.sub(r'\b\w{1,3}\b','', text)

    text = re.sub(' +', ' ',text)

    return text
df = df[['Consumer complaint narrative','Company response to consumer']]

df.head()
df['Company response to consumer'].value_counts()
df['Clean Data'] = df['Consumer complaint narrative'].progress_apply(cleaning)
df1 = df[df['Company response to consumer']=='Closed with explanation']

df1.reset_index(drop = True,inplace = True)

df1.head()
df1.shape
april_prediction = []

for i in tqdm(range(len(df1))):

    b = tokenizer.texts_to_sequences([df1['Clean Data'][i]])

    b_pad = pad_sequences(b,maxlen=300,padding='post',truncating='post')

    results = model.predict(b_pad)

    if results[0][0]>0.5:

        april_prediction.append(1)

    else:

        april_prediction.append(0)
import numpy as np

np.bincount(april_prediction)
april_prediction = []

for i in tqdm(range(len(df1))):

    b = tokenizer.texts_to_sequences([df1['Clean Data'][i]])

    b_pad = pad_sequences(b,maxlen=1000)

    results = model.predict(b_pad)

    if results[0][0]>0.5:

        april_prediction.append(1)

    else:

        april_prediction.append(0)
from collections import Counter

Counter(april_prediction)
df1['Prediction with Test Thres 0.5'] = april_prediction

df1.head()
df1.to_csv('April Closed with explaination Prediction Threshold Test 0.5.csv',index=False)
df2 = df[df['Company response to consumer']!='Closed with explanation']

df2.reset_index(drop = True,inplace = True)

df2.head()
april_prediction_2_classes = []

for i in tqdm(range(len(df2))):

    b = tokenizer.texts_to_sequences([df2['Clean Data'][i]])

    b_pad = pad_sequences(b,maxlen=300,padding='post',truncating='post')

    results = model.predict(b_pad)

    april_prediction_2_classes.append(results[0][0])

#     if results[0][0]>0.5:

#          april_prediction_2_classes.append(1)

#     else:

#          april_prediction_2_classes.append(0)
len(april_prediction_2_classes)
def encoding(text):

    if text == 'Closed with non-monetary relief':

        return int(1)

    elif text == "Closed with monetary relief":

        return int(0)

    

df2['Encoded'] = df2['Company response to consumer'].apply(encoding)

df2['Encoded'] = df2['Encoded'].astype("int64")
df2.head()
prediction = np.array(april_prediction_2_classes)

encoded = np.array(df2['Encoded'].values)

#encoded
len(encoded)
df2.head()
df2['Prob'] = april_prediction_2_classes



def coding(text):

    if text>0.5:

        return 1

    else:

        return 0

    

df2['Prediction'] = df2['Prob'].progress_apply(coding)
encoded = df2['Encoded']

prediction = df2['Prediction']
print(classification_report(encoded,prediction))
print(classification_report(encoded,prediction))
print(classification_report(encoded,prediction))
accuracy_score(encoded,prediction)
import seaborn as sns

sns.heatmap(confusion_matrix(encoded,prediction),annot = True,fmt = 'g')
df2['Probability'] = april_prediction_2_classes
df2.head()
df2.to_csv('Df2.csv',index = False)
april2_prob = []

for i in tqdm(range(len(df2))):

    b = tokenizer.texts_to_sequences([df2['Clean Data'][i]])

    b_pad = pad_sequences(b,maxlen=1000)

    results = model.predict(b_pad)

    april2_prob.append(results[0][0])
recall = []

for thresh in np.arange(0,0.9,0.01):

    thresh = np.round(thresh,2)

    print('Recall Score at threshold {0} is {1}'.format(thresh,recall_score(encoded,(april2_prob>thresh).astype(int))))

    recall.append(recall_score(encoded,(april2_prob>thresh).astype(int)))
precision = []

for thresh in np.arange(0,0.9,0.01):

    thresh = np.round(thresh,2)

    print('Precision Score at threshold {0} is {1}'.format(thresh,precision_score(encoded,(april2_prob>thresh).astype(int))))

    precision.append(precision_score(encoded,(april2_prob>thresh).astype(int)))
f1 = []

for thresh in np.arange(0,0.9,0.01):

    thresh = np.round(thresh,2)

    print('F1 Score at threshold {0} is {1}'.format(thresh,f1_score(encoded,(april2_prob>thresh).astype(int))))

    f1.append(f1_score(encoded,(april2_prob>thresh).astype(int)))
x_axis = range(len(f1))

plt.figure(figsize=(20,12))

plt.plot(x_axis,recall,label = 'Recall')

plt.plot(x_axis,precision,label = 'Precision')

plt.plot(x_axis,f1,label = 'F1 Score')

plt.legend()

plt.grid()



plt.xticks(range(len(f1)))



plt.show()
model.save('Word2vec Monetary and Non Monetary 1000 Length 27K Each.h5')
import pickle



# saving

with open('tokenizer_Word2vec Monetary and Non Monetary 1000 Length 27K Each.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
le.classes_
april_prediction_2_classes = []

for i in tqdm(range(len(df2))):

    b = tokenizer.texts_to_sequences([df2['Clean Data'][i]])

    b_pad = pad_sequences(b,maxlen=1000)

    results = model.predict(b_pad)

    if results[0][0]>0.25:

        april_prediction_2_classes.append(1)

    else:

        april_prediction_2_classes.append(0)
prediction = np.array(april_prediction_2_classes)

encoded = np.array(df2['Encoded'].values)

#encoded
print(classification_report(encoded,prediction))
print(accuracy_score(encoded,prediction))
sns.heatmap(confusion_matrix(encoded,prediction),annot = True,fmt = 'g')
df2['Prediction with OTT Thres 0.25'] = prediction
df2.head()
df2.to_csv('April 2 Classes Prediction Threshold Test 0.5 and OTT Thres 0.25.csv',index=False)
t = 'hello how are you. I am not happy with of Wells Fargo'
b = tokenizer.texts_to_sequences([cleaning(t)])

b_pad = pad_sequences(b,maxlen=300,padding='post',truncating='post')

results = model.predict(b_pad)

results
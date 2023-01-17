# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use("ggplot")



from keras.utils import to_categorical

import keras

from keras.preprocessing.text import one_hot, Tokenizer

from keras.preprocessing.sequence import pad_sequences



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/entity-annotated-corpus/ner.csv", encoding = "ISO-8859-1", error_bad_lines=False)

print('Shape of raw data: ',data.shape)

print(data.columns)

data = data[~(data['sentence_idx'].isnull())]

data.head()
print(data.sentence_idx.max())

print(data.sentence_idx.nunique())
data_sub = data[['sentence_idx','word','tag']]

data_sub.head()
data_sub.drop_duplicates(subset =['sentence_idx','word'], keep = 'first', inplace = True)  #to drop duplicates
a = data_sub.groupby('word',as_index=False).aggregate({'tag':'count'})

b = a.sort_values(['tag'], ascending=[True])

b.reset_index(inplace=True,drop=True)

common_words = list(b[b.tag>5].word) # 10 coud be set as a hyperparameter

print("Total words: ",len(b.word))

print("Number of common words: ",len(common_words))
print('Number of sentences in corpus: ', data.sentence_idx.nunique())

print('Number of unique words in corpus: ', len(data_sub.word.unique()))

print('Number of unique NER tags in corpus: ', len(data_sub.tag.unique()))
set(data_sub.tag)
data_sub[data_sub.word=='.']['tag']
sentences = [] # create empty list of sentences

target = []



num_sen = data.sentence_idx.nunique()

sentence_ids = data.sentence_idx.unique()  #will give non-missing sentence IDs



for i in sentence_ids:

    j=int(i)

    df = data_sub[data_sub.sentence_idx==j]

    a = list(df['word'])

    sentences.append(a)

    a = list(df['tag'])

    target.append(a)
list(zip(sentences[0],target[0]))
plt.hist([len(s) for s in sentences], bins=50)

plt.xlabel('length')

plt.ylabel('#sentences')

plt.show()
def create_unk(sentence):    # takes input 1 sentence in form of words separated in list form

    for i in sentence:

        if i not in common_words and i.isalpha():

            if sentence.index(i)==0 and (i[0].isupper()):

                sentence[sentence.index(i)] = '<UNKSTART>'     #case A

            elif (sentence.index(i)>0) and (i[0].isupper()):

                sentence[sentence.index(i)] = '<UNKMID>'     #case B

            else:

                sentence[sentence.index(i)] = '<UNK>'    #case C

        elif i not in common_words and (i.lower() in common_words):

            ind = sentence.index(i)

            sentence[ind] = i.lower()

            

    return sentence      



create_unk(['This','is','Jatin','saying','YOLO'])   #sample
for j in range(len(sentences)):

    sentences[j] = create_unk(sentences[j])
all_words = list(set(data_sub["word"].values))  

all_words.extend(['<UNKSTART>','<UNKMID>','<UNK>'])



all_tags = list(set(data_sub["tag"].values))  



word2idx = {w: i for i, w in enumerate(all_words)} 

tag2idx = {t: i for i, t in enumerate(all_tags)}  # create dictionary of tags



print(word2idx['main'])

print(tag2idx['B-art'])
print(tag2idx['O'])
max_word_in_sent = max([len(item) for item in sentences])

print('Maximum length of any sentence: ',max_word_in_sent)

X = [[word2idx[w] for w in s] for s in sentences]



X = pad_sequences(maxlen=max_word_in_sent, sequences=X, padding="post",value = word2idx['.'])
X[1]   #sample
y = [[tag2idx[tag] for tag in s] for s in target]

y = pad_sequences(maxlen=max_word_in_sent, sequences=y, padding="post",value = tag2idx['O'])



y = [to_categorical(i, num_classes=len(tag2idx)) for i in y]
print(X[1].shape)

print(y[1].shape)
y[1]  #sample
num_words = len(all_words)

num_tags =  len(all_tags)

embedding_vector_size = 50
t = np.array(y)  # coverting target variable to same data type as of X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X , t, test_size=0.3, random_state=21, shuffle=True)
from keras.models import Model, Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Input,Bidirectional,Dropout,TimeDistributed

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
input = Input(shape=(max_word_in_sent,))

model = Embedding(input_dim=len(all_words), output_dim=embedding_vector_size, input_length=max_word_in_sent)(input)

model = Bidirectional(LSTM(units=32, return_sequences=True))(model)

model = Bidirectional(LSTM(units=10, return_sequences=True))(model)



out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)  # softmax output layer



model = Model(input, out)
model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())



history = model.fit(X_train, y_train, epochs = 7, batch_size =800, validation_data=(X_test, y_test),verbose=1)
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
def check(actual, predictions):

    match = 0

    no_match = 0

    for i in range(len(actual)):

           for j in range(len(actual[i])):  

                if actual[i][j] != tag2idx['O']:   

                    if actual[i][j]==predictions[i][j]:

                        match = match +1

                    else:

                        no_match = no_match + 1

    print("Match: ",match)   

    print("No Match: ",no_match)   

    print("Custom eval Accuracy: ",(match/(match+no_match))*100)
raw_test_predictions = model.predict(X_test)

raw_train_predictions = model.predict(X_train)



test_pred = np.argmax(raw_test_predictions, axis=-1)   #taking index  of maximum probab

train_pred = np.argmax(raw_train_predictions, axis=-1) #taking index  of maximum probab



y_train_labels = np.argmax(y_train, axis=-1)

y_test_labels = np.argmax(y_test, axis=-1)
print("Test set: ")

check(y_test_labels,test_pred)

print("Train set: ")

check(y_train_labels,train_pred)
def unk_sent_preprocess(sentences):

    res = [sub.split() for sub in sentences]

    for i in range(len(res)):

        res[i] = create_unk(res[i])

        

    conv = [[word2idx[w] for w in s] for s in res]

    conv = pad_sequences(maxlen=max_word_in_sent, sequences=conv, padding="post",value = word2idx['.'])

    

    return conv



def unk_sent_predict(pre_pro_sentence):

    

    for i in range(len(pre_pro_sentence)):

    

        p = model.predict(np.array([pre_pro_sentence[i]]))

        p = np.argmax(p, axis=-1)

        print("{:14} ({:5}): {}".format("Word", "True", "Pred"))

        for w,pred in zip(pre_pro_sentence[i],p[0]):

            print("{:14}: {}".format(all_words[w],all_tags[pred]))

    



sent1 = ['Tell me guys is this not a helpful Kaggle kernel','My name is Jatin and I am from India']

a = unk_sent_preprocess(sent1)

unk_sent_predict(a)
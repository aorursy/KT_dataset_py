

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from gensim.models import KeyedVectors

from keras.models import Model

from keras.layers import Input, Dense,Dropout,Embedding,CuDNNGRU,Bidirectional,GlobalMaxPooling1D,GRU

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
embedding = KeyedVectors.load_word2vec_format('../input/quora-insincere-questions-classification/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',binary=True)
train_df = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')

test_df = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
# train_df = train_df.iloc[:50]
from collections import defaultdict

def build_vocab(sentences):

    vocab = defaultdict(int)

    for sentence in sentences:

        for word in sentence.split():

            vocab[word] +=1

#     sorted_d = sorted(vocab.items(), key=lambda x: x[1],reverse=True)

    return vocab
def oov_vocab(vocab,embedding):

    i = 0

    k = 0

    oov = defaultdict(int)

    a = defaultdict(int)

    for word in vocab:

        try:

            a[word] = embedding[word]

            i += vocab[word]

        except:

            oov[word] += vocab[word]

            k +=vocab[word]

    sorted_d = sorted(oov.items(), key=lambda x: x[1],reverse=True)

    print('Found embedding on {:.2%} of vocab'.format((len(a)/len(vocab))))

    print('Found embedding on {:.2%} of all Text'.format(i/(i+k)))

    

    return sorted_d

        
def clean_text1(x):



    x = str(x)

    for punct in "/-":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    return x
def clean_text(x):

    liste = x.split()

    newtext = []

#     print(liste)

    for word in liste:

        

        try:

            

            embedding[word]

            newtext.append(word)

#             print(ckword)

        except KeyError:

            try:

                ckword = word[:-2]+word[-1:]

                embedding[ckword]

                newtext.append(ckword)

#                 print(ckword)

            except KeyError:

                try:

                    ckword = word[:-2]

                    embedding[ckword]

                    newtext.append(ckword)

                except:

                    newtext.append(word)

    return ' '.join(newtext)



import re



def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x
train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_text(x))

train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_numbers(x))

train_df['question_text'] = train_df['question_text'].apply(lambda x: clean_text1(x))
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



## some config values 

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



## fill up the missing values

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((max_features, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    try:

        embedding_vector = embedding[word]

    except:

        embedding_vector = None

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
inp = Input(shape=(maxlen,))

x = Embedding(max_features,embed_size, weights=[embedding_matrix])(inp)

x = Bidirectional(CuDNNGRU(64,return_sequences=True))(x)

# x = GRU(64,return_sequences=True)(x)

x = GlobalMaxPooling1D()(x)

x = Dense(6,activation='relu')(x)

x = Dropout(0.1)(x)

x = Dense(1,activation='sigmoid')(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_paragram_test_y = model.predict(test_X, batch_size=1024, verbose=1)

x = (pred_paragram_test_y>0.5).astype(int)
data = pd.DataFrame({'qid':test_df['qid'].values})
data['prediction'] = x
data.to_csv("submission.csv", index=False)
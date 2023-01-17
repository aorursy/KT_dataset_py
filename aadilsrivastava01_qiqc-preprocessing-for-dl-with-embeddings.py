import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import os

from tqdm import tqdm

tqdm.pandas()

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print(f'Train shape: {df_train.shape}')

print(f'Test shape: {df_test.shape}')
df_train.head()
df = pd.concat([df_train,df_test],sort=True)



df.shape
def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
embedding = load_embed('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
from collections import defaultdict



def build_vocab(sentences):

    fd = defaultdict(int)

    for sentence in tqdm(sentences):

        for word in sentence:

            fd[word]+=1

    return fd    
def embed_intersection(vocab,embedding):

    temp = {}

    oov = {}

    i = 0

    j = 0

    

    for word in vocab.keys():

        try:

            temp[word] = embedding[word]

            i+=vocab[word]

        except:

            oov[word] = vocab[word]

            j+=vocab[word]

            pass

    

    print(f"Found embeddings for {(len(temp)/len(vocab)*100):.3f}% of vocab")

    print(f"Found embeddings for {(i/(i+j))*100:.3f}% of all text")

    

    sorted_x = sorted(oov.items(), key = lambda x: x[1])[::-1]

    return sorted_x
sentences = df['question_text'].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

oov = embed_intersection(vocab,embedding)
oov[0:10]
df['lower_que'] = df['question_text'].apply(lambda x: x.lower())
def fix_case(embedding,vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:

            embedding[word.lower()] = embedding[word]

            count +=1

    print(f'{count} no of words inserted into embedding')
oov = embed_intersection(vocab,embedding)

fix_case(embedding,vocab)

oov = embed_intersection(vocab,embedding)
gc.collect()

oov[0:10]
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 

                       "'cause": "because", "could've": "could have", "couldn't": "could not", 

                       "didn't": "did not",  "doesn't": "does not", "don't": "do not",

                       "hadn't": "had not", "hasn't": "has not", "haven't": "have not", 

                       "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", 

                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 

                       "I'll've": "I will have","I'm": "I am", "I've": "I have", 

                       "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  

                       "i'll've": "i will have","i'm": "i am", "i've": "i have", 

                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 

                       "it'll": "it will", "it'll've": "it will have","it's": "it is", 

                       "let's": "let us", "ma'am": "madam", "mayn't": "may not", 

                       "might've": "might have","mightn't": "might not","mightn't've": "might not have", 

                       "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 

                       "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 

                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 

                       "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 

                       "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",

                       "she's": "she is", "should've": "should have", "shouldn't": "should not", 

                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", 

                       "this's": "this is","that'd": "that would", "that'd've": "that would have", 

                       "that's": "that is", "there'd": "there would", "there'd've": "there would have", 

                       "there's": "there is", "here's": "here is","they'd": "they would", 

                       "they'd've": "they would have", "they'll": "they will", 

                       "they'll've": "they will have", "they're": "they are", "they've": 

                       "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 

                       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",

                       "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 

                       "what'll've": "what will have", "what're": "what are",  "what's": "what is", 

                       "what've": "what have", "when's": "when is", "when've": "when have", 

                       "where'd": "where did", "where's": "where is", "where've": "where have", 

                       "who'll": "who will", "who'll've": "who will have", "who's": "who is",

                       "who've": "who have", "why's": "why is", "why've": "why have", 

                       "will've": "will have", "won't": "will not", "won't've": "will not have",

                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 

                       "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",

                       "y'all're": "you all are","y'all've": "you all have","you'd": "you would", 

                       "you'd've": "you would have", "you'll": "you will", 

                       "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def cont_map(embedding):

    known = []

    for cont in contraction_mapping:

        if cont in embedding:

            known.append(cont)

    return known
cont_map(embedding)
def fix_cont(sentence,mapping):

    sentence = str(sentence)

    specials = ["’", "‘", "´", "`"]

    for each in specials:

        sentence = sentence.replace(each,"'")

    sentence = " ".join([mapping[word] if word in mapping else word for word in sentence.split(" ")])

    return sentence
df['fixed_question'] = df['lower_que'].apply(lambda x: fix_cont(x,contraction_mapping))
sentences = df['fixed_question'].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

oov = embed_intersection(vocab,embedding)
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
def unknown_punct(embed, punct):

    unknown = []

    for p in punct:

        if p not in embed:

            unknown.append(p)

    return unknown
print('Unknown Puctuations')

print(unknown_punct(embedding,punct))
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ",

                 "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", 

                 '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 

                 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 

                 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }



def fix_punt(sentence,punct,mapping):

    for p in mapping:

        sentence = sentence.replace(p, mapping[p])

    

    for p in punct:

        sentence = sentence.replace(p, f' {p} ')

        

    return sentence
df['fixed_question'] = df['fixed_question'].apply(lambda x: fix_punt(x,punct,punct_mapping))



sentences = df['fixed_question'].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

oov = embed_intersection(vocab,embedding)
oov[:50]
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 

                'travelling': 'traveling', 'counselling': 'counseling', 

                'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 

                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 

                'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 

                'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 

                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much',

                'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 

                'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', 

                "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 

                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018',

                'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', 

                "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 

                'demonitization': 'demonetization', 'demonetisation': 'demonetization', 

                'pokémon': 'pokemon','redmi': 'company','oneplus':'company','bhakts':'Worshippers','…': '...'}
def fix_spellings(sentence,mapping):

    for word in mapping.keys():

        sentence = sentence.replace(word,mapping[word])

    return sentence
df['fixed_question'] = df['fixed_question'].apply(lambda x: fix_spellings(x,mispell_dict))



sentences = df['fixed_question'].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

oov = embed_intersection(vocab,embedding)



gc.collect()
df_train['fix_ques'] = df_train['question_text'].apply(lambda x: x.lower())

df_train['fix_ques'] = df_train['fix_ques'].apply(lambda x: fix_cont(x,contraction_mapping))

df_train['fix_ques'] = df_train['fix_ques'].apply(lambda x: fix_punt(x,punct,punct_mapping))

df_train['fix_ques'] = df_train['fix_ques'].apply(lambda x: fix_spellings(x,mispell_dict))



df_test['fix_ques'] = df_test['question_text'].apply(lambda x: x.lower())

df_test['fix_ques'] = df_test['fix_ques'].apply(lambda x: fix_cont(x,contraction_mapping))

df_test['fix_ques'] = df_test['fix_ques'].apply(lambda x: fix_punt(x,punct,punct_mapping))

df_test['fix_ques'] = df_test['fix_ques'].apply(lambda x: fix_spellings(x,mispell_dict))
del df
vocab_size = len(vocab) + 1

max_len = 65
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
def process_data(data):

    t = Tokenizer(filters='')

    t.fit_on_texts(data)

    data = t.texts_to_sequences(data)

    data = pad_sequences(data,maxlen = max_len)

    return data, t.word_index,t
X, word_index, tokenizer = process_data(df_train['fix_ques'])
from sklearn.model_selection import train_test_split



y = df_train['target'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=42)
X_train.shape
def make_embed_mat(embedding,word_index,vocab_size):

    embds = np.stack(embedding.values())

    emb_mean,emb_std = embds.mean(), embds.std()

    embed_size = embds.shape[1]

    word_index = word_index

    embedding_matrix = np.random.normal(emb_mean,emb_std,(vocab_size,embed_size))

    

    for word,i in word_index.items():

        if i>=vocab_size:

            continue

        embedding_vec = embedding.get(word)

        if embedding_vec is not None:

            embedding_matrix[i] = embedding_vec

    return embedding_matrix
embed_matrix = make_embed_mat(embedding,word_index,vocab_size)

del word_index

gc.collect()
embed_matrix.shape
from keras import backend as K



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.layers import Dense, Embedding, CuDNNGRU, Bidirectional, GlobalAveragePooling1D

from keras.layers import GlobalMaxPooling1D,concatenate,Input, Dropout

from keras.optimizers import Adam

from keras.models import Model
def make_model(embedding_matrix, embed_size=300, loss='binary_crossentropy'):

    inp = Input(shape=(max_len,))

    x = Embedding(input_dim=vocab_size,output_dim=embed_size,weights=[embedding_matrix],trainable=False)(inp)

    x = Bidirectional(CuDNNGRU(128,return_sequences=True))(x)

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

    avg_pl = GlobalAveragePooling1D()(x)

    max_pl = GlobalMaxPooling1D()(x)

    concat = concatenate([avg_pl,max_pl])

    dense  = Dense(64, activation="relu")(concat)

    dense   = Dropout(rate = 0.7)(dense)

    output = Dense(1, activation="sigmoid")(dense)

    

    model = Model(inputs=inp, output=output)

    model.compile(loss=loss,optimizer=Adam(lr=0.0001), metrics=['accuracy', f1])

    return model
model = make_model(embed_matrix)
model.summary()
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
checkpoints = ModelCheckpoint('model.h5',monitor='val_f1',mode='max',save_best_only='True',verbose=True)

reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)
epochs = 10

batch_size = 128

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 

                    validation_data=[X_test, y_test], callbacks=[checkpoints, reduce_lr])
import matplotlib.pyplot as plt



plt.figure(figsize=(12,8))

plt.plot(history.history['acc'], label='Train Accuracy')

plt.plot(history.history['val_acc'], label='Test Accuracy')

plt.legend(('Train Acc', 'Val Acc'))

plt.show()
model.load_weights('model.h5')
pred = model.predict(X_test,batch_size=512, verbose=1)
from sklearn.metrics import f1_score



def tweak_threshold(pred, truth):

    thresholds = []

    scores = []

    for thresh in np.arange(0.1, 0.501, 0.01):

        thresh = np.round(thresh, 2)

        thresholds.append(thresh)

        score = f1_score(truth, (pred>thresh).astype(int))

        scores.append(score)

    return np.max(scores), thresholds[np.argmax(scores)]
score_val, threshold_val = tweak_threshold(pred, y_test)



print(f"Scored {round(score_val, 4)} for threshold {threshold_val} with treated texts on validation data")
test = tokenizer.texts_to_sequences(df_test['fix_ques'])

test = pad_sequences(test,maxlen = max_len)



gc.collect()
pred_test = model.predict(test,batch_size=512, verbose=1)
df_test['prediction'] = (pred_test>0.39).astype(int)
df_test.head()
sub = df_test.drop(labels=['question_text','fix_ques'],axis=1)
sub.to_csv(path_or_buf='submission.csv',index=False)
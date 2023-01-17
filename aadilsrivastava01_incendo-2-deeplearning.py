import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/incendo-hackthon/train_dataset.csv')

test = pd.read_csv('../input/incendo-hackthon/test_dataset.csv')
train.head(10)
train.iloc[394]['EssayText']
train.shape
train.dtypes
train.isna().sum()
import matplotlib.pyplot as plt

%matplotlib inline
train['Essayset'].value_counts(normalize=True).plot.bar()
train['max_score'].value_counts().plot.bar()
train.loc[train['Essayset'].isna()==True]
X = train.copy()

X_test = test.copy()
from collections import defaultdict



def count_value(df):

        dic = defaultdict(int)

        dic1 = defaultdict(int)

        df = df.dropna()

        for val in df['Essayset']:

            if val in [1.0,2.0,5.0,6.0]:

                dic[val]+=1

            else:

                dic1[val]+=1

        return dic,dic1
count_value(X)
X.loc[(X['Essayset'].isna() == True) & (X['max_score'] == 3),['Essayset']] = 6.0

X.loc[(X['Essayset'].isna() == True) & (X['max_score'] == 2),['Essayset']] = 8.0
import seaborn as sns

plt.figure(1,figsize=(16,16))





plt.subplot(321)

sns.distplot(X.loc[X['score_3'].isna()!=True,['score_3']])

plt.subplot(322)

sns.boxplot(y=X.loc[X['score_3'].isna()!=True,['score_3']])



plt.subplot(323)

sns.distplot(X.loc[X['score_4'].isna()!=True,['score_4']])

plt.subplot(324)

sns.boxplot(y=X.loc[X['score_4'].isna()!=True,['score_4']])



plt.subplot(325)

sns.distplot(X.loc[X['score_5'].isna()!=True,['score_5']])

plt.subplot(326)

sns.boxplot(y=X.loc[X['score_5'].isna()!=True,['score_5']])



plt.show()
X.loc[X['score_3'].isna()==True,['score_3']] = X['score_3'].mean()

X.loc[X['score_4'].isna()==True,['score_4']] = X['score_4'].mean()

X.loc[X['score_5'].isna()==True,['score_5']] = X['score_4'].mean()
X.isna().sum()
X['score'] = X.loc[:,['score_1','score_2','score_3','score_4','score_5']].mean(axis=1)
X = X.drop(labels = ['score_1','score_2','score_3','score_4','score_5'],axis =1)

X['score'] = X['score'].round()

X.head()
X['score'] = X['score'].astype('category')
X['score'].value_counts().plot.bar()
df = pd.concat([X,X_test],sort=True)
df.shape
from collections import defaultdict

# def build_vocab(df):

#         dic = defaultdict(int)

#         sentences = df['EssayText'].values

#         for sentence in sentences:

#             for word in sentence.split():

#                 dic[word] +=1

#         return dic



def build_vocab(sentences):

        dic = defaultdict(int)

        for sentence in sentences:

            for word in sentence:

                dic[word] +=1

        return dic

sentences = df['EssayText'].apply(lambda x: x.split()).values

vocab = build_vocab(sentences)
len(vocab)
def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index
embedding = load_embed('../input/glove-840b-300d/glove.840B.300d.txt')
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

    

    sorted_x = sorted(oov.items(),key = lambda x: x[1])[::-1]

    return sorted_x
oov = embed_intersection(vocab,embedding)
oov[:10]
df['lower'] = df['EssayText'].apply(lambda x: x.lower())
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
oov[:10]
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
df['fixed'] = df['lower'].apply(lambda x: fix_cont(x,contraction_mapping))
sentences = df['fixed'].apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

oov = embed_intersection(vocab,embedding)
oov[:10]
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
df['fixed'] = df['fixed'].apply(lambda x: fix_punt(x,punct,punct_mapping))



sentences = df['fixed'].apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

oov = embed_intersection(vocab,embedding)
oov[:50]
mispell_dict = {"grna": "RNA", "telephase":"telophase","nucleus2":"nucleus",

               "nuclues":"nucleus","permiable":"permeable","mitocondria":"mitochondria",

                "meosis": "meiosis","nucleas":"nucleus","trna2":"RNA","nucles":"nucleus",

                "inasive":"invasive","strechability":"stretchability","nucleaus":"nucleus",

                "phythons":'pythons',"phython":"python","fluncked":"flunked",

                "expirament":"experiment","memebrane":"membrane","trna3":"RNA",

                "mperature":"temprature","satelittes":"satellites","orginizes":"organizes",

                "obsorbs":"absorbs","membrane2":"membrane","membrane3":"membrane",

                "diffussion":"diffusion","permiability":"permeability","cillia":"cilia",

                "mrna3":"RNA","ribosome2":"ribosome","resperation":"respiration",

                "dna3":"DNA","nuclus":"nucleus","trna2":"RNA","nucles":"nucleus",

                "meosis":"meiosis","nucleas":"nucleus"

               }
def fix_spelling(sentence,mapping):

    for word in mapping.keys():

        sentence = sentence.replace(word,mapping[word])

    return sentence
df['fixed'] = df['fixed'].apply(lambda x: fix_spelling(x,mispell_dict))



sentences = df['fixed'].apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

oov = embed_intersection(vocab,embedding)
X.head()
X['fixed'] = X['EssayText'].apply(lambda x: x.lower())

X['fixed'] = X['fixed'].apply(lambda x: fix_cont(x,contraction_mapping))

X['fixed'] = X['fixed'].apply(lambda x: fix_punt(x,punct,punct_mapping))



X_copy = X.copy()



X['fixed'] = X['fixed'].apply(lambda x: fix_spelling(x,mispell_dict))





X_test['fixed'] = X_test['EssayText'].apply(lambda x: x.lower())

X_test['fixed'] = X_test['fixed'].apply(lambda x: fix_cont(x,contraction_mapping))

X_test['fixed'] = X_test['fixed'].apply(lambda x: fix_punt(x,punct,punct_mapping))

X_test['fixed'] = X_test['fixed'].apply(lambda x: fix_spelling(x,mispell_dict))
del df
vocab_size = len(vocab) + 1

max_len = 50
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
def process_text(data):

    t = Tokenizer(filters='')

    t.fit_on_texts(data)

    data = t.texts_to_sequences(data)

    data = pad_sequences(data,maxlen=max_len)

    return data,t.word_index,t
X_processed,word_index,tokenizer = process_text(X['fixed'])
X_processed.shape
X_processed[0]
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

y = X['score']

y = to_categorical(y)
X_train,X_val,y_train,y_val = train_test_split(X_processed,y,test_size=0.25,random_state=42)
def make_embed_matrix(embedding,word_index,vocab_size):

    embds = np.stack(embedding.values())

    embd_mean,embd_std = embds.mean(),embds.std()

    embed_size = embds.shape[1]

    word_index = word_index

    embedding_matrix = np.random.normal(embd_mean,embd_std,(vocab_size,embed_size))

    

    for word,i in word_index.items():

        if i>=vocab_size:

            continue

        embedding_vec = embedding.get(word)

        if embedding_vec is not None:

            embedding_matrix[i] = embedding_vec

    return embedding_matrix

            
import gc

embed_matrix = make_embed_matrix(embedding,word_index,vocab_size)

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
def make_model(embed_matrix,embed_size=300,loss='categorical_crossentropy'):

    inp = Input(shape=(max_len,))

    x = Embedding(input_dim=vocab_size,output_dim=embed_size,weights=[embed_matrix],trainable=False)(inp)

    x = Bidirectional(CuDNNGRU(128,return_sequences=True))(x)

    avg_pl = GlobalAveragePooling1D()(x)

    max_pl = GlobalMaxPooling1D()(x)

    concat = concatenate([avg_pl,max_pl])

    dense = Dense(128,activation='relu')(concat)

    dense = Dropout(rate = 0.7)(dense)

    output = Dense(4,activation='sigmoid')(dense)

    

    model = Model(input=inp,output=output)

    model.compile(loss=loss,optimizer=Adam(lr=0.0001),metrics=['accuracy', f1])

    return model
model = make_model(embed_matrix)
model.summary()
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
checkpoints = ModelCheckpoint('model.h5',monitor='val_f1',mode='max',save_best_only='True',verbose=True)

reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)
epochs = 100

batch_size = 64
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 

                    validation_data=[X_val, y_val], callbacks=[checkpoints, reduce_lr])
import matplotlib.pyplot as plt



plt.figure(figsize=(12,8))

plt.plot(history.history['acc'], label='Train Accuracy')

plt.plot(history.history['val_acc'], label='Test Accuracy')

plt.legend(('Train Acc', 'Val Acc'))

plt.show()
model.load_weights('model.h5')
submission = tokenizer.texts_to_sequences(X_test['fixed'])

submission = pad_sequences(submission,maxlen=max_len)
pred_sub = model.predict(submission,batch_size=512,verbose=1)



pred_sub = np.argmax(pred_sub,axis=1)
X_test['essay_score'] = pred_sub
X_test.head()
sub = X_test.drop(labels=['min_score','max_score','clarity','coherent','EssayText','fixed'],axis=1)
sub.columns = ['id','essay_set', 'essay_score']



sub.head()
sub.to_csv(path_or_buf = 'submission.csv',index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import operator

import re

import string

import os

from tqdm import tqdm

tqdm.pandas()



from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,SpatialDropout1D

from keras.layers import Bidirectional, GlobalMaxPool1D,GlobalMaxPool2D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
!ls ../input/glovetwitter27b100dtxt
train=pd.read_csv("../input/nlp-getting-started/train.csv")

test=pd.read_csv("../input/nlp-getting-started/test.csv")

sample=pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
train.shape,test.shape
train.head()
## Load the embeddings:



def load_embedding(file):

    def get_coefs(word,*arr):

        return word,np.array(arr,dtype='float32')

    

    embeddings_index=dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)

    

    

    return embeddings_index

    
glove='../input/glovetwitter27b100dtxt/glove.twitter.27B.200d.txt'
embed_glove = load_embedding(glove)
print(len(embed_glove))
def build_vocab(text):

    #sentence=text.apply(lambda x:x.split()).values

    vocab={}

    for s in tqdm(text):

        for word in s:

            try:

                vocab[word]+=1

            except KeyError:

                vocab[word]=1

    return vocab

            
def check_coverage(vocab,embedding_index):

    known_words={}

    unknown_words={}

    nb_known_words=0

    nb_unknown_words=0

    

    for word in tqdm(vocab.keys()):

        try:

            known_words[word]=embedding_index[word]

            nb_known_words+=vocab[word]

        except:

            unknown_words[word]=vocab[word]

            nb_unknown_words+=vocab[word]

            

    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    

    return unknown_words
sentences = train["text"].progress_apply(lambda x: x.split()).values

vocab = build_vocab(sentences)

print({k: vocab[k] for k in list(vocab)[:5]})

## Check coverage:

oov=check_coverage(vocab,embed_glove)
oov[:10]
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def known_contractions(embed):

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known
print("- Known Contractions -")

print("   Glove(Twitter) :")

print(known_contractions(embed_glove))
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
train['text_clean']=train['text'].progress_apply(lambda x:clean_contractions(x,contraction_mapping))

sentence=train['text_clean'].apply(lambda x:x.split())

vocab=build_vocab(sentence)
oov=check_coverage(vocab,embed_glove)
'.' in embed_glove
'2' in embed_glove
## Clean the text

def clean_text(x):



    x = str(x)

    for punct in "/-'":

        x = x.replace(punct, ' ')

    for punct in '&':

        x = x.replace(punct, f' {punct} ')

    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':

        x = x.replace(punct, '')

    

    return x
train['text_clean']=train['text_clean'].progress_apply(lambda x:clean_text(x))

sentence=train['text_clean'].apply(lambda x:x.split())

vocab=build_vocab(sentence)
oov=check_coverage(vocab,embed_glove)
oov[:10]
print('I' in embed_glove)

print('i' in embed_glove)
def clean_numbers(x):



    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    x = re.sub('[0-9]','#',x)

    return x
train['text_clean']=train['text_clean'].progress_apply(lambda x: clean_numbers(x))

train['text_clean']=train['text_clean'].apply(lambda x:x.lower())
sentence=train['text_clean'].apply(lambda x:x.split())

vocab=build_vocab(sentence)

oov=check_coverage(vocab,embed_glove)
oov[:100]
'disease' in embed_glove
mispell_dict={'mh###':'mh','\x89û':'#','\x89ûò':'','#th':'th','prebreak':'pre break','##yr':'year','re\x89û':'re','\x89ûó':'#','bestnaijamade':'best i made','##pm':'pm','diseasese':'disease','diseasese':'disease','udhampur':'city','#km':'km','mediterraneanean':'mediterranean','#x#':'x','qew#c#m#xd':'','rea\x89û':'real','ww#':'w','###pm':'pm','enchanged':'exchanged','#rd':"rd",'bb#':'b','ps#':'ps','bioterrorism':'bio terrorism','bioterror':'bio terror','soudelor':'shoulder','disea':'disease','funtenna':'fun','don\x89ûªt':'do not','crematoria':'crematorium','utc####':'utc','time####':'time','#km':'km','#pm':'pm','\x89ûïwhen':'when','#st':'st','##km':'km','#nd':'nd','#th':'th','##w':'w','irandeal':'iran deal','spos':'spot','mediterran':'mediterranean','inundation':'inundate','it\x89ûªs':'it is','o###':'o','you\x89ûªve':'you have','china\x89ûªs':'china','animalrescue':'animal rescue','canaanites':'canaan','linkury':'malware','###w':'w','##inch':'inch','rì©union':'reunion','mhtw#fnet':'net','microlight':'micro light','i#':'i','##m':'m','a#':'a','lt#':'it','ices\x89û':'ice','djicemoon':'moon','icemoon':'ice moon','be\x89û':'be','ksawlyux##':'as','sinjar':'jar','#wd':'wd','by\x89û':'by','prophetmuhammad':'prophet muhammad','m###':'m','i\x89ûªm':'i am','mikeparractor':'mike','##rd':'rd','dorret':'do ret','##s':'s','read\x89û':'read','x####':'x','encmhz#y##':'enchanged','q#eblokeve':'blok','let\x89ûªs':'let us','can\x89ûªt':'can not','kerricktrial':'trial','twia':'tw','naved':'paved','nasahurricane':'nasa hurricane','vvplfqv##p':'v','pantherattack':'panther attack','youngheroesid':'young hero id','injuryi':'injury','america\x89ûªs':'america','s\x89û':'s','socialnews':'social news','##k':'k','z##':'z','of\x89û':'of','cybksxhf#d':'c','strategicpatience':'strategic patience','sittwe':'sit','here\x89ûªs':'here','summerfate':'summer fate','b#':'b','#am':'am','bb##':'b','usagov':'usa gov','grief\x89ûª':'grief','#ò':'o',

'#th':'th',

're#':'re',

'#ó':'o',

'diseasese':'disease',

'don#ªt':'do not',

'#ïwhen':'when',

'#km':'km',

'mediterraneanean':'mediterranean',

'it#ªs':'it is ',

'#pm':'pm',

'you#ªve':'you have',

'chinaªs':'china',

'be#':'be',

'ices#':'ice',

's#':'s',

'by#':'by',

'iªm':'I am',

'#rd':'rd',

'read#':'read',

'enchanged':'exchanged',

'let#ªs':'let us',

'can#ªt':'can not',

'americaªs':'america',

'of#':'of',

'here#ªs':'here is',

'grief#ª':'grief',

'#÷politics':'politics',

'idfire':'id fire',

'karymsky':'sky',

'rexyy':'rex',

'japìn':'japan',

'#wisenews':'wise news',

'carryi':'carrying',

'offensive#ª':'offensive',

'#÷extremely':'extremely',

'lulgzimbestpicts':'pics',

'ramag':'ram',

'diamondkesawn':'diamond',

'raynbowaffair':'rainbow affair',

'viwxy#xdyk':'#',

'lvlh#w#awo':'#',

'yazidis':'yazid',

'#pcs':'pcs',

'waimate':'mate',

'otrametlife':'#',

'#ïa':'#',

'#aug':'august',

'##jst':'just',

'rqwuoy#fm#':'#',

'ks###':'#',

'##l':'#',

'iªve':'I have',

'unsuckdcmetro':'metro',

'realdonaldtrump':'donald trump',

'abbswinston':'winston',

'moll#vd#yd':'#',

'oppressions':'oppression',

'slanglucci':'slang',

'fettilootch':'#',

'worstsummerjob':'worst summer job',

'#st':'st',

'i##':'#',

'åê':'#',

'åè':'#',

'#ïthe':'the',

'warfighting':'war fighting',

'mitt#\x9d':'#',

'#ïwe':'#',

'michael#sos':'michael',

'kurtschlichter':'#',

'beforeitsnews':'before it is news',

'zujwuiomb':'#',

'lonewolffur':'lone wolf',

'm##':'#',

'localarsonist':'local arson',

'wvj##abgm':'#',

'thoyhrhkfj':'#',

'o##f#cyy#r':'#',

'nnmqlzo#':'#',

'temporary###':'temporary',

'pbban':'#',

'm#':'#',

'votejkt##id':'vote',

'rockyfire':'rock fire',

'throwingknifes':'throwing knife',

'dannyonpc':'danny on picture',

'godslove':'god love',

'ptsdchat':'chat',

'cbcca':'#',

'usnwsgov':'us government',

'metrofmtalk':'metro fm talk',

'#pack':'pack',

'bookboost':'book boost',

'ibooklove':'book love',

'aoms':'#',

'foxysiren':'fox',

'blowmandyup':'blow',

'dehmym#lpk':'#',

'itunesmusic':'i tunes music',

'\x89û÷politics':'politics','viralspell':'viral spell'}
def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x

train['text_clean']=train['text_clean'].progress_apply(lambda x: correct_spelling(x,mispell_dict))

sentence=train['text_clean'].apply(lambda x:x.split())

vocab=build_vocab(sentence)

oov=check_coverage(vocab,embed_glove)

test['text_clean']=test['text'].progress_apply(lambda x:clean_contractions(x,contraction_mapping))

test['text_clean']=test['text_clean'].progress_apply(lambda x:clean_text(x))

test['text_clean']=test['text_clean'].progress_apply(lambda x: clean_numbers(x))

test['text_clean']=test['text_clean'].apply(lambda x:x.lower())

test['text_clean']=test['text_clean'].progress_apply(lambda x: correct_spelling(x,mispell_dict))

sentence=test['text_clean'].apply(lambda x:x.split())

vocab=build_vocab(sentence)

oov=check_coverage(vocab,embed_glove)

train_df,val_df=train_test_split(train,test_size=0.1,random_state=100)
embed_size=300

max_features=20000

maxlen=300

train_X=train_df['text_clean'].values

valid_X=val_df['text_clean'].values

test_X=test['text_clean'].values

tokenizer=Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X=tokenizer.texts_to_sequences(train_X)

valid_X=tokenizer.texts_to_sequences(valid_X)

test_X=tokenizer.texts_to_sequences(test_X)



train_X=pad_sequences(train_X,maxlen=maxlen)

valid_X=pad_sequences(valid_X,maxlen=maxlen)

test_X=pad_sequences(test_X,maxlen=maxlen)



train_Y=train_df['target'].values

valid_Y=val_df['target'].values

train_X.shape,valid_X.shape
all_embs = np.stack(embed_glove.values())

emb_mean,emb_std = all_embs.mean(), all_embs.std()

embed_size = all_embs.shape[1]



word_index = tokenizer.word_index

len(word_index),emb_mean,emb_std,embed_size
nb_words = min(max_features, len(word_index))

embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in tqdm(word_index.items()):

    if i >= max_features: continue

    embedding_vector = embed_glove.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
embedding_matrix.shape
## Simple TF model with bidirectional LSTM

inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(inp)

x=SpatialDropout1D(0.2)(x)

x = Bidirectional(LSTM(128,return_sequences=True))(x)

x=GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(train_X, train_Y, batch_size=64, epochs=5, validation_data=(valid_X, valid_Y))
## Validate with 0.6 as arbitary threshold

pred_glove_val_y = model.predict([valid_X], batch_size=64, verbose=1)

print(f'F1 Score {f1_score(valid_Y,(pred_glove_val_y>0.6).astype(int))}')
pred_glove_test_y = model.predict([test_X], batch_size=1024, verbose=1)
sample.head()
sample['target']=(pred_glove_test_y>0.6).astype(int)
sample.head()
sample['target'].value_counts()
sample.to_csv('sample_submission.csv',index=False)
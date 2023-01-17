import numpy as np 

import pandas as pd 

import emoji

import keras

import seaborn as sns

import matplotlib.pyplot as plt

import string

import re

import tensorflow as tf



from nltk.corpus import stopwords

from nltk.util import ngrams

from nltk.tokenize import word_tokenize



from collections import defaultdict

from collections import Counter



from sklearn import decomposition, model_selection,preprocessing, metrics

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer







train = pd.read_csv("../input/nlp-getting-started/train.csv")

test= pd.read_csv("../input/nlp-getting-started/test.csv")

print(train.head(),train.shape)
plt.hist(train['target'])

train['target'].describe()
DATA_PATH = '../input/spelling/aspell.txt'

misspell_data = pd.read_csv(DATA_PATH,

                            sep=':',

                            names=['correction', 'misspell'])



misspell_data.misspell = misspell_data.misspell.str.strip()

misspell_data.misspell = misspell_data.misspell.str.split(" ")

misspell_data = misspell_data.explode("misspell").reset_index(drop=True)

misspell_data.drop_duplicates("misspell", inplace=True)

miss_corr = dict(zip(misspell_data.misspell, misspell_data.correction))
def misspell_correction(inp):

    for x in inp.split(): 

        if x in miss_corr.keys(): 

            inp = inp.replace(x, miss_corr[x])

    return inp



train["content"] = train["text"].apply(lambda x : misspell_correction(x))

test["content"] = test["text"].apply(lambda x : misspell_correction(x))



print(train["content"].head())

contractions = pd.read_csv("../input/contractions/contractions.csv")

cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))



def cont_to_meaning(val): 

  

    for x in val.split(): 

        if x in cont_dic.keys(): 

            val = val.replace(x, cont_dic[x]) 

    return val



train["content"] = train["content"].apply(lambda x : cont_to_meaning(x))

test["content"] = test["content"].apply(lambda x : cont_to_meaning(x))

print(train["content"])
abbreviations = pd.read_csv("../input/abbreviations-and-slangs-for-text-preprocessing/Abbreviations and Slang.csv")

abrevtn_dic = dict(zip(abbreviations.Abbreviations, abbreviations.Text))



def abbrev2_word(word):

    word= word.lower()

    if word in abrevtn_dic.keys():

        return abrevtn_dic[word]

    else: 

        return word



def abbrev2_text(text):

    sentnc = word_tokenize(text)

    sentnc = [abbrev2_word(word) for word in sentnc]

    text = ' '.join(sentnc)

    return text



train["content"] = train["content"].apply(lambda x: abbrev2_text(x))

test["content"] = test["content"].apply(lambda x: abbrev2_text(x))


train["length"] = train["text"].str.len()

print(train.head())


fig, ax = plt.subplots(figsize=(10,5))

plt.hist(train[train['target']==0]['length'],alpha = 0.6, bins=100, label='Not')

plt.hist(train[train['target']==1]['length'],alpha = 1, bins=100, label='Yes')

ax.set(title= 'length of tweet Vs count for each length',xlabel= 'length',

       ylabel='count',xlim=(0,160))

ax.legend(loc='upper right')

plt.grid()

plt.show()

fig,ax=plt.subplots(figsize=(10,5))



train['word1Len']=train[train['target']==1]['content'].str.count(' ') + 1

train['word0Len']=train[train['target']==0]['content'].str.count(' ') + 1

sns.distplot(train['word1Len'].map(lambda i: np.mean(i)),color='blue')

sns.distplot(train['word0Len'].map(lambda i: np.mean(i)),color='red')

ax.set(xlabel='words in each tweet')

print(train.head())
wordsIn0= []

wordsIn1= []

# collecting all words in each target

for row in train[train['target']==0]['content'].str.split():

        for word in row:

            wordsIn0.append(word)

for row in train[train['target']==1]['content'].str.split():

        for word in row:

            wordsIn1.append(word)

            

words= wordsIn0+wordsIn1
stop = stopwords.words('english')

dic=defaultdict(int)



#collecting stop words used in sentences            

for word in wordsIn0:

    if word in stop:

        dic[word]+=1

top0=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]



for word in wordsIn1:

    if word in stop:

        dic[word]+=1

top1=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 



# plt.rcParams['figure.figsize'] = (18.0, 6.0)

x,y=zip(*top0)

plt.bar(x,y,color='pink',alpha=0.7)

plt.title('Frequent Stopwords in Non didaster tweet')

plt.show()



p,q=zip(*top1)

plt.bar(p,q,color='brown')

plt.title('Frequent Stopwords in disaster tweet')
dic=defaultdict(int)

punc = string.punctuation

for i in wordsIn0:

    if i in punc:

        dic[i]+=1

x,y=zip(*dic.items())



for i in wordsIn1:

    if i in punc:

        dic[i]+=1  

p,q=zip(*dic.items())



plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.bar(x,y)

plt.subplot(122)

plt.bar(p,q)

plt.show()
counter0=Counter(words)

counter1=Counter(wordsIn1)



common0=counter0.most_common()

common1=counter1.most_common()



x0,x1=[],[]

y0,y1=[],[]

for word,count in common0[:40]:

    if (word not in stop) :

        x0.append(word)

        y0.append(count)

for word,count in common1[:40]:

    if (word not in stop) :

        x1.append(word)

        y1.append(count)        

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

sns.barplot(x=y0,y=x0)

plt.subplot(122)

sns.barplot(x=y1,y=x1)

plt.show()
def textBiGrams(line, n=None):

    c_vector = CountVectorizer(ngram_range=(2, 2)).fit(line)

    words_set = c_vector.transform(line)

    freq = words_set.sum(axis=0) 

    words_freq = [(x,freq[0, ind]) for x, ind in c_vector.vocabulary_.items()]

    freq_used =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return freq_used[:n]



plt.figure(figsize=(16,5))

top_bi_grams= textBiGrams(train['content'])[:10]

x,y=map(list,zip(*top_bi_grams))

sns.barplot(x=y,y=x)
def url_html_remove(line):

    url = re.compile(r'https?://\S+|www\.\S+<.*?>')

    x= url.sub(r'',line)

    html=  re.compile('<.*?>')

    y=html.sub(r'',x)

    return y

    

train["clean_content"]=train.content.apply(lambda x: url_html_remove(x))

test["clean_content"]=test.content.apply(lambda x: url_html_remove(x))

def punct_rem(val):   

    for x in string.punctuation: 

        if x in val: 

            val = val.replace(x, " ") 

    return val

train['clean_content']= train['clean_content'].apply(lambda x:' '.join(punct_rem(emoji.demojize(x)).split()))

test['clean_content']= test['clean_content'].apply(lambda x:' '.join(punct_rem(emoji.demojize(x)).split()))



print(train.head())
train['cleaned_words']= train['clean_content'].apply(lambda x:[i for i in x.split() if i not in  stop])



test['cleaned_words']= test['clean_content'].apply(lambda x:[i for i in x.split() if i not in  stop])

print(train['cleaned_words'])
def cv(data):

    count_vectorizer = CountVectorizer()



    emb = count_vectorizer.fit_transform(data)



    return emb, count_vectorizer



word_embeddings={}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f:

    for line in f:

        data=line.split()

        x = data[0]

        vectors=np.asarray(data[1:],'float32')

        word_embeddings[x]=vectors

f.close()
from tqdm import tqdm



corpus= []

for x in tqdm(train["clean_content"]):

    words=[word.lower() for word in word_tokenize(x)]

    corpus.append(words)

for x in tqdm(test["clean_content"]):

    words=[word.lower() for word in word_tokenize(x)]

    corpus.append(words)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



l=50

token=Tokenizer()

token.fit_on_texts(corpus)

seq=token.texts_to_sequences(corpus)



text=pad_sequences(seq,maxlen=l,truncating='post',

                        padding='post')

index=token.word_index

print(len(index))
len_matrix=len(index)+1

word_matrix=np.zeros((len_matrix,100))



for word,i in tqdm(index.items()):

    if i < len_matrix:

        vec=word_embeddings.get(word)

        if vec is not None:

            word_matrix[i]=vec
length= train.shape[0]

print("length:",length)

train_data = text[:length]

test_data=text[length:]



labels = train["target"].values



x_train, x_val, y_train, y_val = train_test_split(train_data,labels, test_size=0.2, 

                                                random_state=10)



print('train data shape', x_train.shape)

from keras.layers import Embedding

from keras.models import Sequential

from keras.layers import Embedding, LSTM,Dense, SpatialDropout1D, Dropout

from keras.initializers import Constant

from keras.optimizers import Adam



model=Sequential()



embedding=Embedding(len_matrix,100,embeddings_initializer=Constant(word_matrix),

                   input_length=l,trainable=False)



model.add(embedding)

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(100, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))





model.add(Dense(1, activation='sigmoid'))

optimzer=Adam(learning_rate=5e-5)



model.compile(loss='binary_crossentropy',optimizer=optimzer,

              metrics=['accuracy'])

model.summary()
lstm_history=model.fit(x_train,y_train,batch_size=16,epochs=15,

                  validation_data=(x_val,y_val),verbose=2)
def show_plots(neural_ntwk):

    loss_vals = neural_ntwk['loss']

    val_loss_vals = neural_ntwk['val_loss']

    epochs = range(1, len(neural_ntwk['accuracy'])+1)

    

    f, ax = plt.subplots(nrows=1,ncols=2,figsize=(16,4))

    

    ax[0].plot(epochs, loss_vals, color='R',marker='o',

               linestyle=' ', label='Train Loss')

    ax[0].plot(epochs, val_loss_vals, color='B',

               marker='*', label='Val Loss')

    ax[0].set(title='Train & Val Loss', xlabel='Epochs',ylabel='Loss')

    ax[0].legend(loc='best')

    ax[0].grid(True)

    

    # plot accuracies

    acc_vals = neural_ntwk['accuracy']

    val_acc_vals = neural_ntwk['val_accuracy']



    ax[1].plot(epochs, acc_vals, color='navy', marker='o',

               ls=' ', label='Train Accuracy')

    ax[1].plot(epochs, val_acc_vals, color='firebrick',

               marker='*', label='Val Accuracy')

    ax[1].set(title='Train & Val Accuracy',xlabel='Epochs',ylabel='Accuracy')

    ax[1].legend(loc='best')

    ax[1].grid(True)

    

    plt.show()

    plt.close()

    

    del loss_vals, val_loss_vals, epochs, acc_vals, val_acc_vals

# show_plots(neural_ntwk1.history)

show_plots(lstm_history.history)
tGloVe = model.predict(test_data)

test_pred_GloVe_int = tGloVe.round().astype('int')

print(test_pred_GloVe_int)

pred= np.concatenate(test_pred_GloVe_int)

print(pred)

ids =test['id']



output = pd.DataFrame({'id':ids,

                      'target': pred})

print(output)

output.to_csv('realfake_pred.csv', index=False)
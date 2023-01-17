# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/movie-review-sentiment-analysis-kernels-only"))
df=pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv",delimiter='\t')
df.head()
test=pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv",delimiter='\t')
test.head()
df.info()
#Unique setneces in our dataset
df['SentenceId'].nunique()
tmp=df.head(50)
tmp
#phrases having only single character
import re
def find_onechar(x):
#     if x.startswith('A'):
    if re.search(r'^\W$',x):
        return(x)
print(tmp['Phrase'].apply(find_onechar).count())
df[df['Phrase']==df['Phrase'].apply(find_onechar)] ## need to delete as no information
test[test['Phrase']==test['Phrase'].apply(find_onechar)]
#Phrases with only one word
def find_oneword(x):
    if re.search(r'^\w+$',x):
        return(x)
    
print('total distinct 1 words in tmp : {} '.format(tmp['Phrase'].apply(find_oneword).nunique()))
print('total distinct 1 words in df : {} '.format(df['Phrase'].apply(find_oneword).nunique()))

#No of words which are lonely present in Phrase
# type(df['Phrase'].apply(find_oneword).value_counts().sort_values(ascending=False).to_frame())
df['Phrase'].apply(find_oneword).value_counts(ascending=False)
tmp_idx=tmp.index[tmp['Phrase']==max(tmp['Phrase'], key=len)]
tmp_idx
tmp.loc[tmp_idx]
#Longest sentence in train df
idx=df.index[df['Phrase']==max(df['Phrase'], key=len)]
print ( 'row index with maximum length of phrase :' ,idx)
# df.loc[idx]['Phrase']
from IPython.display import display
print(max(df['Phrase'], key=len))
val=df.loc[idx].values
val
maxLen = len(max(df['Phrase'], key=len).split())
print('maximum length of Phrase : ', maxLen)
maxLen1 = len(max(test['Phrase'], key=len).split())

idx1=test.index[test['Phrase']==max(test['Phrase'], key=len)]
print('maximum length of Phrase : ', maxLen1)
print ('====================================')

print(max(test['Phrase'], key=len))
print ('====================================')
print ('index of max length in test :', idx1)
print ( test.loc[idx1])
# https://github.com/cryer/Emojify/blob/master/emo_utils.py
def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
import os
print(os.listdir("../input/glove-global-vectors-for-word-representation"))
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt')
len(word_to_index)
word="film"
vec=word_to_vec_map[word]
print('shape of vector is :', vec.shape)
print(vec)
index_to_word[1]
# x1=np.array(["funny lol", "lets play baseball", "I self-glorification u","food is ready for you",])
# # from textblob import TextBlob
# # # x1[0].apply(lambda x: str(TextBlob(x).correct()))
# # # TextBlob("self-glorification").correct()
# sent=[]
# temp=[]
# def list_flatten(l, a=None):
#     #check a
#     if a is None:
#         #initialize with empty list
#         a = []
#     for i in l:
#         if isinstance(i, list):
#             list_flatten(i, a)
#         else:
#             a.append(i)
#     return a

# for i in x1:
#     print(i)
#     sent=i.lower().split()
# #     print(sent)
#     for j in sent :
#         flag=0
#         match=re.search(r'\w-[\w]',j)
#         if match:
#             flag=1
# #             print(i, '====',  j)
# #             sent.remove(j)
#             idx=sent.index(j)
# #             print (idx)
#             temp=[]
#             temp=j.split('-')
# #             print(temp)
#     if flag==1:
#         sent.insert(idx,temp)
#         sent=list_flatten(sent)
    
# #     print(sent)
# #     list(np.array(sent).flat)
#     print(sent)

import re 
x1=np.array(["funny lol", "lets play baseball", "I self-glorification u","food is ready for you",])
for i in x1:
    print(i)
    match=re.search(r'\w-[\w]',i)
    if match:
        i=re.sub(r'(\w)-([\w])',r'\1 \2',i)
        print(i)
x1=np.array(["funny lol", "lets play baseball", "I self@glorification u ", "food is ready for you", "It's there"])

# from nltk.stem import PorterStemmer

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

st = PorterStemmer()
lem = WordNetLemmatizer()

def sentence_to_indices(X,word_to_index,max_len):
    m=X.shape[0]
    fault=[]
    #vecs=word_to_vec_map["film"].shape[0]
    X_indices=np.zeros((m,max_len))
    
    for i in range(m):
        #print(X[i])
        match=re.search(r'\w[!@#$%^&*()_+=-][\w]',X[i])
        if match:
            X[i]=re.sub(r'(\w)[!@#$%^&*()_+=-]([\w])',r'\1 \2',X[i])
#             print(i)
#         print(X[i])
        sentence=X[i].lower().split()
#         print(sentence)
        #print(sentence)
        count=0
        for j in sentence:
            try:
                if j in word_to_index:
                    X_indices[i,count] = word_to_index[j]            
#                 elif word_to_index[str(TextBlob(st.stem(j)).correct())]:
                else:
#                 str(TextBlob(st.stem(j)).correct()) not in word_to_index:
#                 print (j ,'is not in word_to_index. It is in line ', i)            
                    X_indices[i,count] = word_to_index[str(TextBlob(st.stem(j)).correct())]
            except:
                fault.append((j,i))
            count=count+1
    return X_indices,fault

print ('shape is ', x1.shape)
sent,fault=sentence_to_indices(x1,word_to_index,6)
print(sent)
print('=====')
print(fault)
print(fault)
print(  "it's" in word_to_index)
from textblob import Word
print(Word("it's").lemmatize())
# word_to_index[str(TextBlob(st.stem("it's")).correct())]
# print(TextBlob("it's").correct())

# print(word_to_index[str(TextBlob(st.stem("it's")).correct())])
# from nltk.stem import PorterStemmer

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

st = PorterStemmer()
lem = WordNetLemmatizer()

print( st.stem('substitutable'))
print(lem.lemmatize('substitutable'))
# print(word_to_vec_map[st.stem('substitutable')])
from textblob import Word
print(Word('substitutable').lemmatize())
from textblob import TextBlob
print(TextBlob(st.stem('substitutable')).correct())
print(TextBlob('substitut').correct())
from keras.layers import Embedding
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len=len(word_to_index)+1
    vec_length=word_to_vec_map["film"].shape[0]
    emb_matrix=np.zeros((vocab_len,vec_length))
    
    for word,index in word_to_index.items():
        emb_matrix[index,:]=word_to_vec_map[word]
        
    embedding_layer = Embedding(vocab_len,vec_length,trainable=False)
    embedding_layer.build((None,))
    
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
t1=tmp['Sentiment'].head()
t1
from keras.utils.np_utils import to_categorical 
to_categorical(t1,num_classes=4)

df.head()
word_to_index['#']
X= df['Phrase']
Y= df['Sentiment']
print(X.shape)
print(Y.shape)
tmp_train=df['Phrase'].head(170)
# tmp_train
df[(df.index>=151) & (df.index<=170)]
display(df[(df['PhraseId']==105156 ) & (df['SentenceId']==5555)] )#105156, 5555,
tmp_train.count()
tmp_index,fault=sentence_to_indices(tmp_train, word_to_index,max_len= 50)
fault
# str(TextBlob(st.stem('demonstr')).correct())
# word_to_index['demonstr']
# str(TextBlob('demonstr').correct())
print(st.stem('-ERB-'))
print(lem.lemmatize('-ERB-'))
print(word_to_index['demonstrating'] )
print(word_to_index['manipulative'] )

# from nltk.stem import PorterStemmer

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

sts = SnowballStemmer('english')
lem = WordNetLemmatizer()

print( st.stem('-erb-'))
# print(lem.lemmatize('substitutable'))
# # print(word_to_vec_map[st.stem('substitutable')])
# from textblob import Word
# print(Word('substitutable').lemmatize())
# X_train_indices,fault = sentence_to_indices(X_train, word_to_index, 56)
# # # sentence_to_indices(x1,word_to_index,6)
# # # (t1,num_classes=4)
# # Y_train_oh = to_categorical(Y_train, num_classes= 5)
# print(datetime.datetime.now())
# tmp2=df['Phrase'].head(1000000
# tmp2_ind,fault2 = sentence_to_indices(tmp2, word_to_index, 56)
# print(datetime.datetime.now())

# # 156059
# tmp3=[]
# print(datetime.datetime.now())
# tmp3=df.iloc[(df['Phrase'].index.values> 145000)]
# # tmp2_ind,fault2 = sentence_to_indices(tmp2, word_to_index, 56)

# display(tmp3.head())
# print(type(tmp3))
# print('--------')
# display(tmp1.head())
# print(type(tmp1))

# print(datetime.datetime.now())

# tmp4=tmp3['Phrase'].reset_index()['Phrase']
# tmp4.head()
# print(datetime.datetime.now())

# tmp4_ind,fault4 = sentence_to_indices(tmp4, word_to_index, 56)
# print(datetime.datetime.now())

# print(datetime.datetime.now())
# tmp3=df.head()
# # print(tmp3['Phrase'].index.values)
# # tmp3_ind,fault3 = sentence_to_indices(tmp3, word_to_index, 56)
# # print(datetime.datetime.now())

X_train= df.reset_index()['Phrase']
# Y_train= df['Sentiment']
# X_train=X_train.reset_index()['Phrase']
# print(X_train.shape)
# print(Y_train.shape)
X_train.head()
from datetime import datetime
print(datetime.now())
X_train_ind,fault = sentence_to_indices(X_train, word_to_index, 56)
# X_test_ind,fault1 = sentence_to_indices(X_test, word_to_index, 56)
from datetime import datetime
print(datetime.now())
from keras.utils.np_utils import to_categorical 
Y_train= df['Sentiment']
Y_train_ind=to_categorical(Y_train,num_classes=5)
print(Y_train_ind.shape) 
print(X_train_ind.shape)
from sklearn.model_selection import train_test_split

X_to_train,X_to_test,Y_to_train,Y_to_test=train_test_split(X_train_ind,Y_train_ind, test_size=0.2, 
                                                           random_state=22)
print(X_to_train.shape)
print(X_to_test.shape)
print(Y_to_train.shape)
print(Y_to_test.shape)
type(fault)
print(len(fault))
tmp_fault=fault[:10]
print(tmp_fault)
# def Myfun(x):
#     return x[0]
# print(sorted(tmp_fault))
# print(sorted(tmp_fault,key=Myfun))
fault1=[x[0] for x in fault]
fault_out=list(set(sorted(fault1)))
print(fault_out)
import numpy as np
# np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
# np.random.seed(1)
def Sentiment(input_shape, word_to_vec_map, word_to_index):
    """
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape, dtype='int32')
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)   
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(5)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=[sentence_indices],outputs=X)
    
    ### END CODE HERE ###
    
    return model
maxLen=56
model = Sentiment((maxLen,), word_to_vec_map, word_to_index)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(datetime.now())
model.fit(X_to_train, Y_to_train, epochs = 10, batch_size = 32, shuffle=True)
print(datetime.now())
print(X_to_test.shape)
print(Y_to_test.shape)
print(datetime.now())
pred=model.predict(X_to_test) 
print(datetime.now())
print(datetime.now())
loss, acc = model.evaluate(X_to_test, Y_to_test)
print('LOSS is :- ' , loss)
print('ACCURACY is :- ', acc)
print(datetime.now())
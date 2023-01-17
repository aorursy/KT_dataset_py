# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("/kaggle/working"))

# Any results you write to the current directory are saved as output.

#dataset for this tutoria : http://mlg.ucd.ie/datasets/bbc.html

#didn't use the load function the point behind this is to pickel data for next traitement
import  pickle,os

dir='../input/my-data'
def save(dir):
    #result_file=open(dir+'/result.i2','wb')        
    list=os.listdir(dir) 
    obj=[list]   
    for file in list:
        #print (file)
        f=open(dir+"/"+file,'rb')  
        obj.append(f.read())
        #obj=[f.read(),f]
        
    pickle.dump(obj,open('/kaggle/working/result.pkl','wb'),2)

def load(path):
    f=open(path+"/"+'result.pkl','rb')


    obj=pickle.load(f)    
    for i in range(1,len(obj)):
        file=open(path+"/"+obj[0][i-1],'wb')
        file.writelines(obj[i])
        file.close()

#os.chdir("/kaggle/working/")
save('../input/')
#converting into pickel
favorite_color = pickle.load( open( "/kaggle/working/result.pkl", "rb" ) )
#deleting the cell where names of files are in
del favorite_color[0] 
#partitionniting between headers and description , separartion using \\n\\n
head=[]
des=[]
for l in favorite_color:
    o=str(l).partition('\\n\\n')
    head.append(o[0])
    des.append(o[2])
    
    
#the data type still be bytes to convert it into string i should have used     
FN = 'vocabulary-embedding'
seed=42
vocab_size = 40000
embedding_dim = 100
lower = False # dont lower case the text
from collections import Counter
from itertools import chain
def get_vocab(lst):
    vocabcount = Counter(w for txt in lst for w in txt.split())
    vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
    return vocab, vocabcount
vocab, vocabcount = get_vocab(head+des)
print(list(vocab))
#transforemer les donner de map vers lite parceque map n'est pas iterable
l_vocab=[]
for ch in vocab:
    print(ch)
    l_vocab.append(ch)
len(l_vocab)

import matplotlib.pyplot as plt
%matplotlib inline
plt.plot([vocabcount[w] for w in l_vocab]);
plt.gca().set_xscale("log", nonposx='clip')
plt.gca().set_yscale("log", nonposy='clip')
plt.title('word distribution in headlines and discription')
plt.xlabel('rank')
plt.ylabel('total appearances');
empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word
def get_idx(vocab, vocabcount):
    word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    
    idx2word = dict((idx,word) for word,idx in word2idx.items())

    return word2idx, idx2word
word2idx, idx2word = get_idx(l_vocab, vocabcount)
word2idx
idx2word
from keras.utils.data_utils import get_file
fname = 'glove.6B.%dd.txt'%embedding_dim
import os
datadir_base = os.path.expanduser(os.path.join('~', '.keras'))
if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
datadir = os.path.join(datadir_base, 'datasets')
glove_name = os.path.join(datadir, fname)
if not os.path.exists(glove_name):
    path = 'glove.6B.zip'
    path = get_file(path, origin="http://nlp.stanford.edu/data/glove.6B.zip")
    !unzip {datadir}/{path}
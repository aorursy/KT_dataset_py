import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

from collections import Counter

import re

import nltk 

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer

from keras.models import Sequential

from keras.layers import Dense

from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass

# /kaggle/input/movie-review/movie_reviews/movie_reviews/neg/cv537_13516.txt



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
negdirectory='/kaggle/input/movie-review/movie_reviews/movie_reviews/neg/'

posdirectory='/kaggle/input/movie-review/movie_reviews/movie_reviews/pos/'
# Load documnet into memory 

def load_doc(filename):

    file=open(filename,'r')   # Open the file as read only              

    txt=file.read()  # Read all text

    file.close()  # close the file 

    return txt   # return txt

# To clean documents

def clean_doc(doc):

    tokens=doc.split() # split into tokens by white space

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))  # prepare regex for char filtering

    tokens = [re_punc.sub('', w) for w in tokens] # remove punctuation from each word

    tokens = [word for word in tokens if word.isalpha()] # remove remaining tokens that are not alphabetic

    stop_words = set(stopwords.words('english')) # filter out stop words

    tokens = [w for w in tokens if not w in stop_words] # filter out short tokens

    tokens = [word for word in tokens if len(word) > 5]

    return tokens 

# to save documents 

def doc_save(fileanme,lines):

    data='\n'.join(lines)

    file=open(fileanme,'w')

    file.write(data)

    file.close()

def add_doc_vocab(filename,vocab):

    filecontent=load_doc(filename)

    token=clean_doc(filecontent)

    vocab.update(token)

def process_doc(directory,vocab):

    for file in os.listdir(directory):

        if not file.endswith('.txt'):

            next

        if file.startswith('cv9'):

            continue 

        path=directory+file

        add_doc_vocab(path,vocab)
# Create Vocabulary  

vocab=Counter()

process_doc(negdirectory,vocab)

process_doc(posdirectory,vocab)

#print(len(vocab))

#print(vocab.most_common(100))

min_occurance=5

tokens=[k for k,c in vocab.items() if c>=min_occurance]

doc_save('tokens.txt',tokens)
def doc_to_lines(filename,vocab):

    filecontent=load_doc(filename)

    token=clean_doc(filecontent)

    token=[t for t in token if t in vocab]

    return ' '.join(token)

def process_to_all_doc(directory,vocab,is_train):

    lines=list()

    for file in os.listdir(directory):

        if not file.endswith('.txt'):

            next 

        if is_train and file.startswith('cv9'):

            continue

        if not is_train and not file.startswith('cv9'):

            continue 

        line=doc_to_lines(directory+file,vocab) 

        lines.append(line)

    return lines

def crate_tokenizer(lines):

    tokenizer=Tokenizer()

    tokenizer.fit_on_texts(lines)

    return tokenizer


tokens=load_doc('tokens.txt')

tokens=set(tokens.split())



# For Train 

neg=process_to_all_doc(negdirectory,tokens,True)

pos=process_to_all_doc(posdirectory,tokens,True)

tain_doc=neg+pos

tain_lables= [0 for _ in range(len(neg))]+ [1 for _ in range(len(pos))]



# For test 

test_neg=process_to_all_doc(negdirectory,tokens,False)

test_pos=process_to_all_doc(posdirectory,tokens,False)

test_doc=test_neg+test_pos

test_lables= [0 for _ in range(len(test_neg))]+ [1 for _ in range(len(test_pos))]    
tokenizer=crate_tokenizer(tain_doc)

#x_train=Tokenizer.texts_to_matrix(tain_doc,mode='freq')

Xtrain = tokenizer.texts_to_matrix(tain_doc, mode='freq')

Xtest = tokenizer.texts_to_matrix(test_doc, mode='freq')
print('Xtrain.shape',Xtrain.shape)

print('Xtest.shape',Xtest.shape)

Ytrain=np.array(tain_lables)

Ytest=np.array(test_lables)

print('tain_lables shape',np.array(Ytrain).shape)

print('test_lables shape',np.array(Ytest).shape)
in_shape=Xtrain.shape[1]

def Create_model(in_shape):

    model=Sequential()

    model.add(Dense(50,input_shape=(in_shape,),activation='relu'))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.summary()

    plot_model(model,show_shapes=True)

    return model

def evaluate_model(x_train,y_train,x_test,y_test):

    scorelits=list()

    model=Create_model(in_shape)

    model.fit(x_train,y_train,epochs=10,verbose=2)

    _,score=model.evaluate(x_test,y_test,verbose=0)

    scorelits.append(score)

    return scorelits

def prepare_data(train_doc,test_doc,mode):

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(train_doc)

    Xtrain = tokenizer.texts_to_matrix(train_doc, mode=mode)

    Xtest = tokenizer.texts_to_matrix(test_doc, mode=mode)

    return Xtrain, Xtest    
modes = ['binary', 'count', 'tfidf', 'freq']

results = pd.DataFrame()

for mod in modes:

    Xtrain, Xtest=prepare_data(tain_doc,test_doc,mod)

    results[mod] = evaluate_model(Xtrain, Ytrain, Xtest, Ytest)

results.describe()
results.boxplot()

plt.show()
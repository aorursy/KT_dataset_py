# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import bz2
import csv 

train_file = bz2.BZ2File('/kaggle/input/amazonreviews/train.ft.txt.bz2')
#test_file = bz2.BZ2File("/kaggle/input/amazonreviews/test.ft.txt.bz2")
train_data = train_file.readlines()
#test_data = test_file.readlines()
del train_file
#del test_file
train_data = [x.decode('utf-8') for x in train_data]
#test_data = [x.decode('utf-8') for x in test_data]
#print the first elemnt of the list obtained 
train_data[0:5]
#Separaing the target and feature
train_labels=[]
train_text=[]
for x in train_data[1:]:
    train_labels.append((x[0:10]))
    train_text.append(x[10:].strip())
    

print("Shape of the train data labels",len(train_labels))
#print("Shape of the train data text",len(train_text))
print("Shape of the test data labels",len(test_labels))
#print("Shape of the test data text",len(test_text))
del train_data
#del test_data
### Get the Length of each line and find the maximum length

def len_sentence(data):
    maxlen = 0
    for i in range(len(data)):
        length = len(data[i])
        if(length > maxlen):
            maxlen = length
    print("Max length of sentence in the document is ",maxlen) 
### Set Different Parameters for the model
MAX_FEATURES = 10000
MAX_NUM_WORDS = 1015
EMBEDDING_SIZE = 200
#Import the Tokenizer from keras preprocessing text
from keras.preprocessing.text import Tokenizer
## Apply Tokenizer
#Initialize the Tokenizer class with maximum vocabulary count as MAX_NB_WORDS initialized at the start of step2.
t = Tokenizer(num_words=MAX_FEATURES,filters= '!"#$%&()*+,-./:;<=>?@[\]^_`{|}\n“~')
#Now, using fit_on_texts() from Tokenizer class, lets encode the data
t.fit_on_texts(train_text)
print(t.document_count)
print(len(t.word_index))
train_text = t.texts_to_sequences(train_text)
#test_text = t.texts_to_sequences(test_text)
from tensorflow.keras.preprocessing.sequence import pad_sequences
X_train = pad_sequences(train_text, maxlen = MAX_NUM_WORDS)
#X_test= pad_sequences(test_test, maxlen = MAX_NUM_WORDS)
y_train = np.asarray([0 if x.split(' ')[0] == '__label__1' else 1 for x in train_labels])
#y_test = np.asarray([0 if x.split(' ')[0] == '__label__1' else 1 for x in test_labels])
del train_text,train_labels
del test_text ,test_labels












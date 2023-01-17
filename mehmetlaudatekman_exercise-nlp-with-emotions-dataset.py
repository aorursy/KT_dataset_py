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
test = pd.read_csv('/kaggle/input/emotions-dataset-for-nlp/test.txt',sep=";",header=None)

train = pd.read_csv('/kaggle/input/emotions-dataset-for-nlp/train.txt',sep=";",header=None)

val = pd.read_csv('/kaggle/input/emotions-dataset-for-nlp/val.txt',sep=";",header=None)
train.head()
test.head()
val.head()
train.info()
test.info()
val.info()
# Splitting X and Y

x_train = train.iloc[:,0] 

y_train = train.iloc[:,1] 



x_test = test.iloc[:,0] 

y_test = test.iloc[:,1] 



x_val = test.iloc[:,0] 

y_val = test.iloc[:,1] 



x_train,y_train = np.array(x_train),np.array(y_train)

x_test,y_test = np.array(x_test),np.array(y_test)

x_val,y_val = np.array(x_val),np.array(y_val)



print(x_train.shape,"|",y_train.shape)

print(x_test.shape,"|",y_test.shape)

print(x_val.shape,"|",y_val.shape)
print(train.iloc[:,1].value_counts())



int_y_train = []

int_y_test = []

int_y_val = []



for l in y_train:

    

    if l == "joy":        

        int_y_train.append(0)

        

    if l == "sadness":       

        int_y_train.append(1)            

    if l == "anger":      

        int_y_train.append(2)      

    if l == "fear":

        int_y_train.append(3)

    if l == "love":

        int_y_train.append(4)

    if l == "surprise":

        int_y_train.append(5)

        

        

for l in y_test:

    

    if l == "joy":        

        int_y_test.append(0)

        

    if l == "sadness":       

        int_y_test.append(1)            

    if l == "anger":      

        int_y_test.append(2)      

    if l == "fear":

        int_y_test.append(3)

    if l == "love":

        int_y_test.append(4)

    if l == "surprise":

        int_y_test.append(5)

        

for l in y_val:

    

    if l == "joy":        

        int_y_val.append(0)

        

    if l == "sadness":       

        int_y_val.append(1)            

    if l == "anger":      

        int_y_val.append(2)      

    if l == "fear":

        int_y_val.append(3)

    if l == "love":

        int_y_val.append(4)

    if l == "surprise":

        int_y_val.append(5)

        





        

int_y_train,int_y_test,int_y_val = np.array(int_y_train),np.array(int_y_test),np.array(int_y_val)

from sklearn import preprocessing

from keras.utils import np_utils



le = preprocessing.LabelEncoder()

le.fit(int_y_train)



encoded_y_train = le.transform(int_y_train)

encoded_y_test = le.transform(int_y_test)

encoded_y_val = le.transform(int_y_val)



encoded_y_train = np_utils.to_categorical(encoded_y_train)

encoded_y_test = np_utils.to_categorical(encoded_y_test)

encoded_y_val = np_utils.to_categorical(encoded_y_val)



print(encoded_y_train)



print(int_y_train[:10])
x_train[0]
x_test[1]
x_val[1]
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords



stopwords = stopwords.words('english')



x_train_cl = []

x_test_cl = []

x_val_cl = []





# Deleting stopwords

for text in x_train:

    

    text = text.split()

    text = [word for word in text if word not in stopwords]

    text = " ".join(text)

    x_train_cl.append(text)

    

for text in x_test:

    

    text = text.split()

    text = [word for word in text if word not in stopwords]

    text = " ".join(text)

    x_test_cl.append(text)



for text in x_val:

    

    text = text.split()

    text = [word for word in text if word not in stopwords]

    text = " ".join(text)

    x_val_cl.append(text)

    

x_train,x_test,x_val = np.array(x_train_cl),np.array(x_test_cl),np.array(x_val_cl)



# We use total_text for fitting tokenizer in general 

total_text = np.concatenate((x_train,x_test,x_val),axis=0)



num_words = 2000



tokenizer = Tokenizer(num_words = num_words) 



tokenizer.fit_on_texts(total_text)



list(tokenizer.word_index)[:10]
# Tokenizing everything



x_train_token = tokenizer.texts_to_sequences(x_train)

x_test_token = tokenizer.texts_to_sequences(x_test)

x_val_token = tokenizer.texts_to_sequences(x_val)



total_token = np.concatenate((x_train_token,x_test_token,x_val_token),axis=0)
# Padding

print(np.mean([len(text) for text in total_token]))
x_train_pad = pad_sequences(x_train_token,20) 

x_test_pad = pad_sequences(x_test_token,20)

x_val_pad = pad_sequences(x_val_token,20)



print(x_train_pad[0],end="\n-------------------------------------------------------------------------\n")

print(x_train_pad[1])
from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense,CuDNNGRU,Embedding,Bidirectional





model = Sequential()



model.add(Embedding(input_dim=2000

                   ,output_dim=100

                   ,input_length=20))



model.add(Bidirectional(CuDNNGRU(units=16,return_sequences=True)))



model.add(Bidirectional(CuDNNGRU(units=8)))



model.add(Dense(6,activation="softmax"))



model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])



model.summary()
print(x_train_pad.shape)

print(y_train.shape)

model.fit(x_train_pad,encoded_y_train,epochs=10,batch_size=20)
preds = model.predict_classes(x_test_pad)

from sklearn.metrics import accuracy_score



accuracy_score(preds,int_y_test)
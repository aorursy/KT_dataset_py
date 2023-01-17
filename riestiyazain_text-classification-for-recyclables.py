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
train_filename = '/kaggle/input/recyclables-data/train_label.csv'

test_filename = '/kaggle/input/recyclables-data/test_label.csv'
#Reading train data

train_df= pd.read_csv(train_filename).values

train_data = train_df[:,0]

train_label = train_df[:,1]



#Reading test data 

test_df = pd.read_csv(test_filename).values

test_data = test_df[:,0]

test_label = test_df[:,1]



print(train_data[:2])

print(train_label[:2])

print(test_data[:2])

print(test_label[:2])
#Convert train and test label to hot vector format



from keras.utils import to_categorical



train_label = to_categorical(train_label)

test_label = to_categorical(test_label)

print(train_label.shape)

print(test_label.shape)

print(train_label)

print(test_label)

#Tokenizer 



from keras_preprocessing.text import Tokenizer

#initializing the tokenizer

tokenizer = Tokenizer()

#fit tokenizer only on train data

tokenizer.fit_on_texts(train_data)

#train data tokenization

train_seq = tokenizer.texts_to_sequences(train_data)

train_matrix = tokenizer.sequences_to_matrix(train_seq,mode='tfidf')

#test data tokenizaion

test_seq = tokenizer.texts_to_sequences(test_data)

test_matrix = tokenizer.sequences_to_matrix(test_seq,mode='tfidf')



print(train_matrix.shape)

print(test_matrix.shape)

print(test_matrix)
from keras import models

from keras import layers





_,num_features = train_matrix.shape

model = models.Sequential()

model.add(layers.Dense(32,activation='relu',input_shape = (num_features,)))

model.add(layers.Flatten())

model.add(layers.Dense(6, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy',

             metrics=['accuracy'])

model.summary()
#callback for early stopping



import keras



callback_es = keras.callbacks.EarlyStopping(monitor='val_loss',

                                           min_delta=0,

                                           patience=4,

                                           verbose=1, mode='auto')
#model fitting



history = model.fit(train_matrix,train_label, epochs = 100,

                   batch_size=2, validation_split=0.2,

                   callbacks=[callback_es])



results = model.evaluate(test_matrix, test_label)

print("Hasil  [loss,acc] untuk data test:")

print(results)
# save and load model and tokenizer



import pickle

model.save('recyclables_text_class.h5')

with open('tokenizer.pickle', 'wb') as handle:

    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
from keras.models import load_model

import pickle

model = load_model('./recyclables_text_class.h5')

with open('./tokenizer.pickle', 'rb') as handle:

    tokenizer = pickle.load(handle)



s  = ['ini bootol aqua bekas','bir bintang','umplungan','pelastik sasetan',

     'kantong belanja','gelas pelastik']

seq_str = tokenizer.texts_to_sequences(s)

enc_str = tokenizer.sequences_to_matrix(seq_str,mode="tfidf")

enc_str.shape

pred = model.predict_classes(enc_str)

print("Prediksi kelas string ' {} ' adalah {}".format(s,pred))
def category(i):

    switcher = {

        0: "plastic_bottle",

        1: "glass_bottle",

        2: "alumunium_can",

        3: "plastic_sachet",

        4: "plastic_bag",

        5: "plastic_pp"

    }

    return switcher.get(i,'invalid category')
print('original category','  ','Predicted category')

print()

for i in range(pred.size):

    print(category(test_label[i].argmax()),'     ',category(pred[i]))
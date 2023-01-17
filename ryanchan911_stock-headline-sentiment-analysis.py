# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/selective-stock-headlines-sentiment/Project6500.xlsx')

df['datetime'] = pd.to_datetime(df['datetime'])

df['headline'][2]
state = 1

df = df.sample(frac=1,random_state=state)

print(df.shape)

df.head()
stop =['i', 'me', 'my', 'myself',

 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',

 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 

 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 

 'being', 'a', 'an', 'the', 'and', 'to', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain','at','of','for','in','v'] 
# Data Preprocessing

df['headline_mod'] = df['headline']

df['headline_mod'] = df['headline_mod'].replace(to_replace='\@+[a-zA-Z]+', value='', regex=True).replace(to_replace='\#+[a-zA-Z]+', value='', regex=True).replace(to_replace='[a-zA-Z]+\…', value='', regex=True).replace(to_replace='…', value='', regex=False).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df = df[['datetime', 'headline', 'headline_mod', 'ticker','sentiment']]

df.drop_duplicates(keep ='first',inplace=True)

df.reset_index(inplace=True, drop=True) 
df.head()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

top_word = 10000

max_words = 50

splitting_num = 8000

tok = Tokenizer(num_words=top_word)

tok.fit_on_texts(df['headline_mod'][:splitting_num])
#Experiment cond.

X_train = tok.texts_to_sequences(df['headline_mod'][:splitting_num])

X_test = tok.texts_to_sequences(df['headline_mod'][splitting_num:])



#Data to be prediceted

Y_train = df['sentiment'][:splitting_num]

Y_test = df['sentiment'][splitting_num:]
from tensorflow.keras.utils import to_categorical

# One-hot category

Y_train = to_categorical(Y_train)

Y_test = to_categorical(Y_test)
X_train = sequence.pad_sequences(X_train, maxlen=max_words)

X_test  = sequence.pad_sequences(X_test,  maxlen=max_words)

print("X_train.shape: ", X_train.shape)

print("X_test.shape: ", X_test.shape)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D
#Model 1 

def model_1():

    model = Sequential()

    model.add(Embedding(top_word, 32, input_length=max_words))

    model.add(Dropout(0.5))

    model.add(LSTM(64,return_sequences = True))

    model.add(LSTM(64))

    model.add(Dropout(0.25))

    model.add(Dense(128, activation="relu"))

    model.add(Dropout(0.25))

    model.add(Dense(2, activation="softmax"))

    

    return(model)



#Model 2

def model_2():

    model = Sequential()

    model.add(Embedding(top_word, 32, input_length=max_words))

    model.add(Dropout(0.5))

    model.add(Conv1D(filters=32, kernel_size=3,padding='same',activation='relu'))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Dropout(0.25))

    model.add(LSTM(64,return_sequences = True))

    model.add(LSTM(64))

    model.add(Dropout(0.25))

    model.add(Dense(128, activation="relu"))

    model.add(Dropout(0.25))

    model.add(Dense(2, activation="softmax"))

    

    return(model)
def model_setup(model,X_train,Y_train,X_test, Y_test):

    #Model setting up

    model.summary()

    # Model compiling

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Model training

    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=2)

    # Model Evaluation

    loss, accuracy = model.evaluate(X_test, Y_test,verbose=0)

    print("-----------------------------------------")

    print("Accuracy of the Training Dataset = {:.2f}".format(accuracy))

    print("Report End")

    return(history)
def result_eva (loss,val_loss,acc,val_acc):

    import matplotlib.pyplot as plt

    %matplotlib inline

    

    epochs = range(1,len(loss)+1)

    plt.plot(epochs, loss,'b-', label ='Training Loss')

    plt.plot(epochs, val_loss,'r--', label ='Validation Loss')

    plt.title("Training and Validation Loss")

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()

    

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, "b-", label="Training Acc")

    plt.plot(epochs, val_acc, "r--", label="Validation Acc")

    plt.title("Training and Validation Accuracy")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.show()
model = model_1()

history = model_setup(model,X_train,Y_train,X_test, Y_test)

result_eva(history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy'])
#Model Prediction

Y_pred = model.predict_classes(X_test,batch_size=10,verbose=2)

Y_target = df['sentiment'][splitting_num:].astype(int)

#print(Y_pred)

#print(Y_target)

tb = pd.crosstab(Y_target,Y_pred,rownames=['label'],colnames=['predict'])

print(tb)
model2 = model_2()

history2 = model_setup(model2,X_train,Y_train,X_test, Y_test)

result_eva(history2.history['loss'], history2.history['val_loss'], history2.history['accuracy'], history2.history['val_accuracy'])
#Model Prediction

Y_pred = model2.predict_classes(X_test,batch_size=10,verbose=2)

Y_target = df['sentiment'][splitting_num:].astype(int)

tb = pd.crosstab(Y_target,Y_pred,rownames=['label'],colnames=['predict'])

print(tb)
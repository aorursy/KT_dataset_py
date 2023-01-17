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
import numpy as np 

import pandas as pd

import pickle

import os

import datetime

from tqdm import tqdm

from statistics import mean 

import matplotlib.pyplot as plt 

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, TimeDistributed, Input,concatenate,Layer

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import EarlyStopping

from keras.models import Sequential

import scipy.stats as stats

from scipy.stats import pearsonr

from scipy.stats import spearmanr

from sklearn.metrics import mean_squared_error



duration_list=[]

batch_sizes=[]

epochs_list=[]

optimizer_list=[]

training_loss_list=[]

test_loss_list=[]

pearson_list=[]

spearman_list=[]



with open('../input/audiolstm/audio_featDict.pkl', 'rb') as f:

    audio_featDict=pickle.load(f)

    

with open('../input/audiolstm/audio_featDictMark2.pkl', 'rb') as f:

    audio_featDictMark2=pickle.load(f)

    

traindf= pd.read_csv("../input/audiolstm/train_split2.csv")

testdf=pd.read_csv("../input/audiolstm/test_split2.csv")

valdf=pd.read_csv("../input/audiolstm/val_split2.csv")

    

with open('../input/audiolstm/AvgAudioFeat2.pkl', 'rb') as f:

    AvgAudioFeat=pickle.load(f)

    

with open('../input/audiolstm/glove_train.pkl', 'rb') as f:

    text_train=pickle.load(f)

    

with open('../input/audiolstm/glove_test.pkl', 'rb') as f:

    text_test=pickle.load(f)

    

with open('../input/audiolstm/glove_val.pkl', 'rb') as f:

    text_val=pickle.load(f)

    

maxlen=520



for key,val in text_train.items():

    text_train[key]=np.pad(val,((0,maxlen-len(val)),(0,0)), 'constant')

    

for key,val in text_test.items():

    text_test[key]=np.pad(val,((0,maxlen-len(val)),(0,0)), 'constant')

    

for key,val in text_val.items():

    text_val[key]=np.pad(val,((0,maxlen-len(val)),(0,0)), 'constant')



    

error=[]

error_text=[]



def ModifyData(df,text_dict):

    X_audio=[]

    X_text=[]

    y_3days=[]

    y_7days=[]

    y_15days=[]

    y_30days=[]



    for index,row in df.iterrows():

        

        try:

            X_text.append(text_dict[row['text_file_name'][:-9]])

        except:

            error_text.append(row['text_file_name'][:-9])

        

        try:

             X_audio.append(np.nan_to_num(AvgAudioFeat[row['text_file_name'][:-9]]))

            

        except:

            Padded=np.zeros(26, dtype=np.float64)

            X_audio.append(Padded)

            error.append(row['text_file_name'][:-9])

            



        y_3days.append(float(row['future_3']))

        y_7days.append(float(row['future_7']))

        y_15days.append(float(row['future_15']))

        y_30days.append(float(row['future_30']))

        

    X_audio=np.array(X_audio)

    X_text=np.array(X_text)

    y_3days=np.array(y_3days)

    y_7days=np.array(y_7days)

    y_15days=np.array(y_15days)

    y_30days=np.array(y_30days)

    



        

    return X_audio,X_text,y_3days,y_7days,y_15days,y_30days







X_train_audio,X_train_text,y_train3days, y_train7days, y_train15days, y_train30days=ModifyData(traindf,text_train)



X_test_audio,X_test_text, y_test3days, y_test7days, y_test15days, y_test30days=ModifyData(testdf,text_test)



X_val_audio,X_val_text,y_val3days, y_val7days, y_val15days, y_val30days=ModifyData(valdf,text_val)



# input_audio_shape = (X_train_audio.shape[1],)

input_text_shape = (X_train_text.shape[1],X_train_text.shape[2])



X_train_audio=np.expand_dims(X_train_audio, axis=2)

X_test_audio=np.expand_dims(X_test_audio, axis=2)

X_val_audio=np.expand_dims(X_val_audio, axis=2)



input_audio_shape=(X_train_audio.shape[1],X_train_audio.shape[2])
X_test_text.shape
def train_text(t_bilstm1,t_fc1,y_train, y_val, y_test, batch_size, epochs, learning_rate):

    

    model_text=Sequential()

    model_text.add(Masking(mask_value=0,input_shape = input_text_shape))

    model_text.add(Bidirectional(LSTM(units=t_bilstm1, dropout=0.8, recurrent_dropout=0.8, activation='tanh' ,return_sequences=True)))

    model_text.add(Dropout(0.8))

    model_text.add(TimeDistributed(Dense(units=t_fc1,activation='relu')))

    model_text.add(Bidirectional(LSTM(units=t_bilstm1,dropout=0, recurrent_dropout=0,activation='tanh')))

    model_text.add(Dense(50,activation='tanh'))

    model_text.add(Dense(1,activation='linear'))

    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model_text.compile( optimizer=adam,loss='mean_squared_error')

    

    

    history=model_text.fit(

    X_train_text,

    y_train,

    batch_size=batch_size,

    epochs=epochs,

    validation_data=(X_val_text , y_val)

    )

    

    

    

    test_loss_text = model_text.evaluate(X_test_text,y_test,batch_size=batch_size)

    train_loss_text = model_text.evaluate(X_train_text,y_train,batch_size=batch_size)

    

    print()

    print("Text Train loss  : {train_loss}".format(train_loss = train_loss_text))

    print("Text Test loss : {test_loss}".format(test_loss = test_loss_text))

    

    print()

    y_pred_text = model_text.predict(X_test_text)

    



    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Text Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='upper left')

    

    save_path = "epochs="+str(epochs)+"_learning-rate"+str(learning_rate)

    save_pkl="y_pred_"+save_path+".pkl"

    

    with open(save_pkl,'wb') as f:

        pickle.dump(y_pred_text,f)

        

    model_text.save(save_path+"_model.h5")

    plt.show()

    plt.savefig(save_path+".png")

    plt.close()

    

    print("Text model trained!")

    

    return y_pred_text
def train_audio(y_train, y_val, y_test, batch_size, epochs, learning_rate):  

    

    #Audio Model

    model_audio=Sequential()

    model_audio.add(Conv1D(filters=128,

               kernel_size=5,

               padding='valid',

               activation='relu',

               input_shape=(26,1)))

    

    

    model_audio.add(MaxPooling1D())



    model_audio.add(Dropout(0.5))

    model_audio.add(Flatten())

    model_audio.add(Dense(300,activation='relu'))

    model_audio.add(Dense(1,activation='linear'))

    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model_audio.compile( optimizer=adam,loss='mean_squared_error')

    

    

    

    history=model_audio.fit(

    X_train_audio,

    y_train,

    batch_size=batch_size,

    epochs=epochs,

    validation_data=(X_val_audio , y_val)

    )



    print("Audio model trained!")

   

    test_loss_audio = model_audio.evaluate(X_test_audio,y_test,batch_size=batch_size)

    train_loss_audio = model_audio.evaluate(X_train_audio,y_train,batch_size=batch_size)

    

    print()

    print("Audio Train loss  : {train_loss}".format(train_loss = train_loss_audio))

    print("Audio Test loss : {test_loss}".format(test_loss = test_loss_audio))

    

    print()

    y_pred_audio = model_audio.predict(X_test_audio)

    



    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Audio Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='upper left')

    

    save_path = "epochs="+str(epochs)+"_learning-rate"+str(learning_rate)

    save_pkl="y_pred_"+save_path+".pkl"

    

    with open(save_pkl,'wb') as f:

        pickle.dump(y_pred_audio,f)

        

    model_audio.save(save_path+"_model.h5")

    plt.show()

    plt.savefig(save_path+".png")

    plt.close()

    

    return y_pred_audio
def combined(duration,y_pred_audio,y_pred_text,y_test):

    

    alpha_range=[0.2,0.4,0.6,0.8]

    for alpha in alpha_range:

        y_pred_combined=y_pred_text*(alpha)+(1-alpha)*y_pred_audio

        

        save_path="CNN_LateFusion_duration_{dur}_alpha_{alph}.pkl".format(dur=duration,alph=alpha)

        

        pickle.dump(y_pred_combined, open(save_path, 'wb'))

        

        mse = mean_squared_error(y_test, y_pred_combined)

        print("MSE for alpha={alpha_val}:".format(alpha_val=alpha)+" "+str(mse))

    return
y_pred_text3days=train_text(t_bilstm1=100,t_fc1=100,y_train=y_train3days, y_val=y_val3days, y_test=y_test3days, batch_size=32, epochs=50, learning_rate=0.001)
y_pred_text7days=train_text(t_bilstm1=100,t_fc1=100,y_train=y_train7days, y_val=y_val7days, y_test=y_test7days, batch_size=32, epochs=50, learning_rate=0.001)
y_pred_text15days=train_text(t_bilstm1=100,t_fc1=100,y_train=y_train15days, y_val=y_val15days, y_test=y_test15days, batch_size=32, epochs=50, learning_rate=0.001)
y_pred_text30days=train_text(t_bilstm1=100,t_fc1=100,y_train=y_train30days, y_val=y_val30days, y_test=y_test30days, batch_size=32, epochs=50, learning_rate=0.001)
y_pred_audio3days=train_audio(y_train=y_train3days, y_val=y_val3days, y_test=y_test3days,batch_size=8, epochs=50, learning_rate=0.001)
y_pred_audio7days=train_audio(y_train=y_train7days, y_val=y_val7days, y_test=y_test7days,batch_size=8, epochs=50, learning_rate=0.001)
y_pred_audio15days=train_audio(y_train=y_train15days, y_val=y_val15days, y_test=y_test15days,batch_size=8, epochs=50, learning_rate=0.001)
y_pred_audio30days=train_audio(y_train=y_train30days, y_val=y_val30days, y_test=y_test30days,batch_size=8, epochs=50, learning_rate=0.001)
combined(3,y_pred_audio3days,y_pred_text3days,y_test3days)
combined(7,y_pred_audio7days,y_pred_text7days,y_test7days)
combined(15,y_pred_audio15days,y_pred_text15days,y_test15days)
combined(30,y_pred_audio30days,y_pred_text30days,y_test30days)
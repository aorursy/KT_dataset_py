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

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, TimeDistributed, Input,concatenate

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import EarlyStopping

from keras.models import Sequential

import scipy.stats as stats

from scipy.stats import pearsonr

from scipy.stats import spearmanr

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder 

from keras.utils import to_categorical

from sklearn.metrics import f1_score



# le = preprocessing.LabelEncoder()

# le.classes_ = np.load('../input/audiolstm/le_BSH.npy')



# onehotencoder = OneHotEncoder() 



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

    

    

traindf= pd.read_csv("../input/audiolstm/train_split_BSH.csv")

testdf=pd.read_csv("../input/audiolstm/test_split_BSH.csv")

valdf=pd.read_csv("../input/audiolstm/val_split_BSH.csv")



    

with open('../input/audiolstm/mittens_train.pkl', 'rb') as f:

    text_train=pickle.load(f)

    

with open('../input/audiolstm/mittens_test.pkl', 'rb') as f:

    text_test=pickle.load(f)

    

with open('../input/audiolstm/mittens_val.pkl', 'rb') as f:

    text_val=pickle.load(f)

    



    



error=[]

error_text=[]





def get_label(inp):

    if inp=="Buy":

        return [1,0,0]

    elif inp=="Hold":

        return [0,1,0]

    elif inp=="Sell":

        return [0,0,1]

    



def change_df(df):

    df['BSH_day3'] = df.apply(lambda row : get_label(row.BSH_day3), axis = 1)

    df['BSH_day7'] = df.apply(lambda row : get_label(row.BSH_day7), axis = 1)

    df['BSH_day15'] = df.apply(lambda row : get_label(row.BSH_day15), axis = 1)

    df['BSH_day30'] = df.apply(lambda row : get_label(row.BSH_day30), axis = 1) 

    return df

    



train_df = change_df(traindf)

val_df = change_df(valdf)

test_df = change_df(testdf)

    

    



def ModifyData(df,text_dict):

    X=[]

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



        lstm_matrix_temp = np.zeros((520, 26), dtype=np.float64)

        i=0

        

        try:

            speaker_list=list(audio_featDict[row['text_file_name'][:-9]])

            speaker_list=sorted(speaker_list, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))

            for sent in speaker_list:

                lstm_matrix_temp[i, :]=audio_featDict[row['text_file_name'][:-9]][sent]+audio_featDictMark2[row['text_file_name'][:-9]][sent]

                i+=1

            X.append(lstm_matrix_temp)



        except:

            Padded=np.zeros((520, 26), dtype=np.float64)

            X.append(Padded)

            error.append(row['text_file_name'][:-9])

            

        

        y_3days.append((row['BSH_day3']))

        y_7days.append((row['BSH_day7']))

        y_15days.append((row['BSH_day15']))

        y_30days.append((row['BSH_day30']))

        

    X=np.array(X)

    X_text=np.array(X_text)

    X=np.nan_to_num(X)

    

    y_3days=np.array(y_3days)

    y_7days=np.array(y_7days)

    y_15days=np.array(y_15days)

    y_30days=np.array(y_30days)

    

        

    return X,X_text,y_3days,y_7days,y_15days,y_30days







X_train_audio,X_train_text,y_train3days, y_train7days, y_train15days, y_train30days=ModifyData(traindf,text_train)



X_test_audio,X_test_text, y_test3days, y_test7days, y_test15days, y_test30days=ModifyData(testdf,text_test)



X_val_audio,X_val_text,y_val3days, y_val7days, y_val15days, y_val30days=ModifyData(valdf,text_val)







input_audio_shape = (X_train_audio.shape[1], X_train_audio.shape[2])

input_text_shape = (X_train_text.shape[1],X_train_text.shape[2])

error
print(X_train_audio.shape)

print(X_train_text.shape)
print(y_train3days.shape)

print(y_test3days.shape)
def get_labels(inp):

    labels = np.argmax(inp,axis=1)

    return labels
def train(duration,t_bilstm1,fc1,a_bilstm1, c_bilstm, c_fc ,y_train, y_val, y_test, batch_size, epochs, learning_rate):

    

    a_fc1=fc1

    t_fc1=fc1

    

    input_text = Input(shape=input_text_shape) 

    input_audio = Input(shape=input_audio_shape)

    

    #Text Model

    

    mask_text=Masking(mask_value=0)(input_text)

    T_bilstm1=Bidirectional(LSTM(units=t_bilstm1, dropout=0.7, recurrent_dropout=0.7, activation='tanh' ,return_sequences=True))(mask_text)

    T_drop1=Dropout(0.7)(T_bilstm1)

    T_fc1=TimeDistributed(Dense(units=t_fc1,activation='relu'))(T_drop1)

    

    

    mask_audio=Masking(mask_value=0)(input_audio)

    A_bilstm1=Bidirectional(LSTM(units=a_bilstm1, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))(mask_audio)

    A_drop1=Dropout(0.5)(A_bilstm1)

    A_fc1=TimeDistributed(Dense(units=a_fc1,activation='relu'))(A_drop1)

    

    

    combined = concatenate([T_fc1, A_fc1])

    

    

    C_bilstm=Bidirectional(LSTM(units=c_bilstm, dropout=0, recurrent_dropout=0, activation='tanh'))(combined)

    C_fc=Dense(units=c_fc,activation='tanh')(C_bilstm)

    C_final=Dense(3, activation='softmax')(C_fc)

    

    model = Model(inputs=[input_text,input_audio], outputs=C_final)

    

    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile( optimizer=adam,loss='categorical_crossentropy')

    

    history=model.fit(

                      [X_train_text,X_train_audio],

                      y_train,batch_size=batch_size,

                      epochs=epochs,

                      validation_data=([X_val_text,X_val_audio], y_val)

                     )

    

   

    test_pred = model.predict([X_test_text,X_test_audio])

    train_pred = model.predict([X_train_text,X_train_audio])

    

    test_pred_labels = get_labels(test_pred)

    train_pred_labels = get_labels(train_pred)



    y_test_labels = get_labels(y_test)

    y_train_labels = get_labels(y_train)

    

    train_f1 = f1_score(y_train_labels,train_pred_labels,average = 'weighted')

    test_f1 = f1_score(y_test_labels,test_pred_labels,average = 'weighted')

    

    print("Train F1 for {duration} days : {train_loss}".format(duration = duration,train_loss = train_f1))

    print("Test F1 for {duration} days : {test_loss}".format(duration = duration,test_loss = test_f1))

    

    

    save_path ="duration_"+str(duration) +"epochs="+str(epochs)+"_learning-rate"+str(learning_rate)

    

    save_pkl="y_pred_"+save_path+".pkl"

    

    with open(save_pkl,'wb') as f:

        pickle.dump(test_pred,f)

    

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc='upper left')

    

    

    

    plt.savefig(save_path+".png")

    plt.close()

    

    model.save(save_path+"_model.h5")

    

    model.save_weights("model_wts"+save_path+"_model.h5")

    



    return
train(duration=3,t_bilstm1=100,fc1=100,a_bilstm1=100,c_bilstm=100,c_fc=50 ,y_train=y_train3days, y_val=y_val3days, y_test=y_test3days, batch_size=64, epochs=60, learning_rate=0.001)
train(duration=7,t_bilstm1=100,fc1=100,a_bilstm1=100,c_bilstm=100,c_fc=50 ,y_train=y_train7days, y_val=y_val7days, y_test=y_test7days, batch_size=64, epochs=60, learning_rate=0.001)
train(duration=15,t_bilstm1=300,fc1=100,a_bilstm1=300,c_bilstm=300,c_fc=50 ,y_train=y_train15days, y_val=y_val15days, y_test=y_test15days, batch_size=64, epochs=60, learning_rate=0.001)
train(duration=30,t_bilstm1=100,fc1=50,a_bilstm1=100,c_bilstm=100,c_fc=100 ,y_train=y_train30days, y_val=y_val30days, y_test=y_test30days, batch_size=64, epochs=60, learning_rate=0.001)
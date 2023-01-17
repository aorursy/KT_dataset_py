import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



import os



from statistics import mean



import random

from tqdm import tqdm_notebook as tqdm

from tqdm import tqdm_pandas



import warnings

warnings.filterwarnings('ignore')
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
import keras

from keras import backend as K

from keras.models import Sequential



from keras.layers import Conv1D,MaxPooling1D,GlobalAveragePooling1D,Flatten,AveragePooling1D

from keras.layers.core import Dense, Dropout

from keras.optimizers import Adam
model = Sequential()



model.add(Conv1D(filters=4 , kernel_size=16, strides=8, activation='relu', input_shape=(150000, 2)))

model.add(Conv1D(filters=4, kernel_size=8, activation='relu'))

model.add(MaxPooling1D(3))



model.add(Conv1D(filters=8, kernel_size=8, activation='relu'))

model.add(Conv1D(filters=8, kernel_size=8, activation='relu'))

model.add(MaxPooling1D(3))



model.add(Conv1D(filters=16, kernel_size=8, activation='relu'))

model.add(Conv1D(filters=16, kernel_size=8, activation='relu'))

model.add(MaxPooling1D(3))



model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))

model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))

model.add(MaxPooling1D(3))



model.add(Flatten())

model.add(Dropout(rate=0.4))

model.add(Dense(1, activation='linear'))



print(model.summary())
model.compile(loss='mean_absolute_error',optimizer='adam')
# scaling the acousting signals



def prepareAd(x):

    x = np.sign(x)*np.log(1 + np.abs(x))/8.7

    #x = np.sign(x)*np.log(1 + np.sqrt(np.abs(x)))/4.4

    return x
def getTrainBatch(dfl):

    batch_size = 1024

    

    x = np.empty([batch_size,150000,2])

    y = np.empty([batch_size,1])

    

    for i,rn in enumerate(np.random.randint(dfl.shape[0]-150000, size=batch_size)):

        df = dfl.loc[rn:rn+149999,:]

        x[i,:,:] = df.loc[:,['acoustic_data_p','acoustic_data_n']].values

        y[i] = df.time_to_failure.values[-1]



    return(x,y)
srows = [5656574,50085878,104677356,138772453,187641820,218652630,245829585,307838917,

 338276287,375377848,419368880,461811623,495800225,528777115,585568144]



nrows = [44429304,54591478,34095097,48869367,31010810,27176955,62009332,30437370,

         37101561,43991032,42442743,33988602,32976890,56791029,36417529]



loss = []

val_loss = [] 



for epoch in range(10):

    for i, (s,n) in enumerate(zip(srows,nrows)):

        print('epoch : ' , epoch,'\t file chunck :',i, end = '\t')



        train_df = pd.read_csv("../input/train.csv",

                           skiprows = s,

                           nrows = n,

                          )

        train_df.columns = ['acoustic_data','time_to_failure']



        print('  max_time_to_failure : ',np.round(train_df.time_to_failure.values[0],2) , end = '\t')



        # scaling

        train_df.acoustic_data = prepareAd(train_df.acoustic_data.values)

        train_df.time_to_failure = train_df.time_to_failure



        # two series

        train_df['acoustic_data_p'] = np.where(train_df['acoustic_data']>=0, np.abs(train_df['acoustic_data']), 0)

        train_df['acoustic_data_n'] = np.where(train_df['acoustic_data']<0, np.abs(train_df['acoustic_data']), 0)



        x_train,y_train = getTrainBatch(train_df)



        history = model.fit(x_train,

                         y_train,

                         batch_size=16,

                         epochs=10,

                         validation_split=0.1,

                         verbose=0)

        

        loss = loss + history.history['loss']

        val_loss = val_loss + history.history['val_loss']

        

        print('  loss : ',np.round(mean(loss[-150:]),2), '\t val_loss : ', np.round(mean(val_loss[-150:]),2))

        

        gc.collect()
# predicting the submission

def predictSubmission(seg_id):

    test_df = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    test_df.acoustic_data =prepareAd(test_df.acoustic_data.values) 



    # two series

    test_df['acoustic_data_p'] = np.where(test_df['acoustic_data']>=0, np.abs(test_df['acoustic_data']), 0)

    test_df['acoustic_data_n'] = np.where(test_df['acoustic_data']<0, np.abs(test_df['acoustic_data']), 0)



    # reshaping

    x_test = test_df.loc[:,['acoustic_data_p','acoustic_data_n']].values.reshape(-1,150000,2)

    

    y = model.predict(x_test)

    return y[0][0]
tqdm_pandas(tqdm())

submission = pd.read_csv('../input/sample_submission.csv')

submission.loc[:,'time_to_failure']=submission.loc[:,'seg_id'].progress_apply(predictSubmission)
submission.describe()
submission.loc[submission.time_to_failure <0,'time_to_failure'] = 0
submission.to_csv('submission_15.csv',index=False)
from IPython.display import FileLink

FileLink('submission_15.csv')
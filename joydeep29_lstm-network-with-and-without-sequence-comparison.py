# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.models import Model

from keras.layers import LSTM,GRU,Dense,Flatten,Input

from keras.activations import sigmoid,relu

from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



csvfile = pd.read_csv('../input/corn_OHLC2013-2017.txt',header=None)

csvfile.columns = [['Date', ' Open Price', ' High Price', ' Low Price', ' Close Price']]



csvfile['Deviation'] = csvfile[' Close Price'] - csvfile[' Open Price']

csvfile['Difference'] = csvfile[' High Price'] - csvfile[' Low Price']



Attribute_of_interest = ['Date',' Close Price','Deviation','Difference']

dataset = csvfile[Attribute_of_interest]





df = csvfile[[' Close Price','Deviation','Difference']]



scalerfordata = MinMaxScaler(feature_range=(0,1))

scalerforoutput = MinMaxScaler(feature_range=(0,1))



transformed_df = scalerfordata.fit_transform(df)

transformed_Y = scalerforoutput.fit_transform(pd.DataFrame(df[' Close Price']))





class SequneceLSTMStateless:

    def __init__(self,batchsize,datatrainX,output_dimension):

        self.batch = batchsize

        self.X = datatrainX

        self.outdim = output_dimension





    def __buildModel__(self):

        ip = Input(batch_shape=(self.batch,self.X.shape[1],self.X.shape[2]))

        FirstLayer = LSTM(2,name = 'LSTMLayer',return_sequences=True,stateful=False)(ip)

        FirstLayer = Flatten()(FirstLayer)

        OutLayer = Dense(self.outdim,activation=relu)(FirstLayer)

        model = Model(input = [ip],output = [OutLayer])

        model.compile(optimizer=keras.optimizers.RMSprop(lr=0.01),loss='mean_squared_error')

        return model



class SequnecelessLSTMStateless:

    def __init__(self,batchsize,datatrainX,output_dimension):

        self.batch = batchsize

        self.X = datatrainX

        self.outdim = output_dimension





    def __buildModel__(self):

        ip = Input(batch_shape=(self.batch,self.X.shape[1],self.X.shape[2]))

        FirstLayer = LSTM(2,name = 'LSTMLayer',return_sequences=False,stateful=False)(ip)

        OutLayer = Dense(self.outdim,activation=relu)(FirstLayer)

        model = Model(input = [ip],output = [OutLayer])

        model.compile(optimizer=keras.optimizers.RMSprop(lr=0.01),loss='mean_squared_error')

        return model

class sequential_data_preparation:

    def __init__(self):

        print("Sequential_data_preparation_started")



    def data_prep_for_keras(self,traindatax, outdata, look_back=7, look_forward_period=1):

        datatrainX, datatrainY = [], []

        for i in range(0, (traindatax.shape[0] - look_back - 1)):

            a = np.array(traindatax)[i:(i + look_back), :]

            datatrainX.append(a)

            datatrainY.append(np.array(outdata)[(i + look_back):(i + look_back + look_forward_period), :])

        return np.array(datatrainX), np.array(datatrainY)





data_prep = sequential_data_preparation()





datatrainX,datatrainY = data_prep.data_prep_for_keras(traindatax=transformed_df,outdata=transformed_Y)

datatrainY = datatrainY.flatten().reshape(datatrainY.shape[0],1)

print(datatrainX.shape)

print(datatrainY.shape)



trainingX,validationX = datatrainX[:200,:,:],datatrainX[200:240,:,:]

trainingY,validationY = datatrainY[:200,:],datatrainY[200:240,:]



model2 = SequnecelessLSTMStateless(batchsize=4,datatrainX=trainingX,output_dimension=trainingY.shape[1])

LSTM_1 = model2.__buildModel__()



model3 = SequneceLSTMStateless(batchsize=4,datatrainX=trainingX,output_dimension=trainingY.shape[1])

LSTM_2 = model3.__buildModel__()





class Histories(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.losses = []



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        self.losses.append(logs.get('loss'))

        return



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return







losslist1=Histories()

losslist2=Histories()



LSTM_1.fit(trainingX,trainingY,batch_size=4,verbose=2,epochs=10,validation_data=[validationX,validationY],callbacks=[losslist1])





LSTM_2.fit(trainingX,trainingY,batch_size=4,verbose=2,epochs=10,validation_data=[validationX,validationY],callbacks=[losslist2])





Loss1 = pd.DataFrame(losslist1.losses,columns=['LSTMWithSequenceTrain'])

Loss2 = pd.DataFrame(losslist2.losses,columns=['LSTMWithoutSequenceTrain'])



TotalLoss = pd.concat([Loss1,Loss2],axis=1)





predicted1 = scalerforoutput.inverse_transform(LSTM_1.predict(validationX,batch_size=4))

predicted2 = scalerforoutput.inverse_transform(LSTM_2.predict(validationX,batch_size=4))



ALLPredict=pd.concat([pd.DataFrame(predicted1,columns=['LSTMWithSequencePredict']),

                      pd.DataFrame(predicted2, columns=['LSTMWithoutSequencePredict'])],axis=1)



Actual = csvfile.iloc[(csvfile.shape[0] - validationY.shape[0]):,:][['Date',' Close Price']]

Actual.index = ALLPredict.index



Comparison = pd.concat([Actual,ALLPredict],axis=1)

plots = Comparison.plot()

plots.get_figure().savefig("Comparison.png")









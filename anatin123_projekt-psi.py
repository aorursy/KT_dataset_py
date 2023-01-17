import numpy as np 

import time, datetime

import pandas as pd

import os

import random

import seaborn as sns

from sklearn import preprocessing

from keras.optimizers import Adam

from keras.models import Sequential

from keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization, Flatten

from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')
def ekg_converter(value):

    return (((value/(2**6)-0.5)*3.3)/1100)*1000

    

def gsr_converter(value):

    return (((value/2**6)*3.3)/0.132)/1000000

    

def time_diff(timestamp, start_time):

    if  start_time-timestamp > 8000:

        return True

    else:

        return False

    

    

def add_data(bit,proc, iter):

    if iter not in [4,10,16,18,41,49]:

        f = open(f'../input/bitalino/bitalino/{bit[1+iter]}', 'r')

        for i in range(2):

            l = f.readline()



        indx = l.find("time")

        t = l[indx+8:indx+17]

        t = [float(val) for val in t.split(':')]

        date = bit[1+iter][-23:-13]



        secs = time.mktime(datetime.datetime.strptime(date, "%Y-%m-%d").timetuple())

        secs += t[0]*60*60

        secs += t[1]*60

        secs += t[2]



        df1 = pd.read_csv(f'../input/procedura/procedura/{proc[iter]}', delimiter='\t', header=None)

        df1 = df1[[6,7,8,9]]

        df1[8] = df1[8].replace('None', np.nan)

        df1 = df1.dropna(axis=0, subset=[8])



        df1[6] = df1[6].replace('emospace1', np.nan)

        df1 = df1.dropna(axis=0, subset=[6])

        df1[8] = df1[8].astype(float)

        df1 = df1[df1[8]>=0.1]



        times = df1[9]

        

        if time_diff(times.iloc[0],secs):

            secs -= 14400

        else:

            secs -= 7200



        times -= secs

        times = times.astype(int)





        df1['ID'] = df1.reset_index().index



        df2 = pd.read_csv(f'../input/bitalino/bitalino/{bit[1+iter]}', delimiter='\t', skiprows=[0, 1, 2], header=None)

        df2 = df2[[5, 6]]

        df2.rename(columns={5: 'EKG', 6: 'GSR'}, inplace = True)



        df2['EKG'] = df2['EKG'].apply(ekg_converter)

        df2['GSR'] = df2['GSR'].apply(gsr_converter)



        df2_new = df2.rolling(10).mean().iloc[::10]



        df2_new['EKG'] = df2_new['EKG'] - df2_new['EKG'].iloc[:100].mean()

        df2_new['GSR'] = df2_new['GSR'] - df2_new['GSR'].iloc[:100].mean()

        df2_new['EKG'] = preprocessing.scale(df2_new['EKG'].values)

        df2_new['GSR'] = preprocessing.scale(df2_new['GSR'].values)



        global data

    

        for i, element in enumerate(times):

            data['EKG_u'].append(df2_new.iloc[element*100-15:element*100+1500,0].values)

            data['GSR'].append(df2_new.iloc[element*100-15:element*100+1500,1].values)

            data['E'].append(df1.iloc[i,1])


data = {'EKG_u':[], 'GSR':[], 'E':[]}

bit = sorted(os.listdir("../input/bitalino/bitalino"))

proc = sorted(os.listdir("../input/procedura/procedura"))

for i in range(0,30,1):

    add_data(bit,proc,i)
seqs = [[data['EKG_u'][i], data['GSR'][i], int(float(data['E'][i]))] for i in range(len(data['E']))]

np.random.shuffle(seqs)



# normal_data = {"1":[], "2":[], "3":[], "4":[], "5":[]}



# for ekg, gsr, emotion in sequence:

#     if emotion == 1:

#         normal_data["1"].append([ekg,gsr,emotion])

#     elif emotion == 2:

#         normal_data["2"].append([ekg,gsr,emotion])

#     elif emotion == 3:

#         normal_data["3"].append([ekg,gsr,emotion])

#     elif emotion == 4:

#         normal_data["4"].append([ekg,gsr,emotion])

#     elif emotion == 5:

#         normal_data["5"].append([ekg,gsr,emotion])

# lowest = min(len(normal_data["1"]),len(normal_data["2"]),len(normal_data["3"]),len(normal_data["4"]),len(normal_data["5"])) + 10



# for i in range(1,6):

#     normal_data[f"{i}"] = normal_data[f"{i}"][:lowest]



# seqs = normal_data["1"]+normal_data["2"]+normal_data["3"]+normal_data["4"]+normal_data["5"]

# random.shuffle(seqs)



X_tmp = []

y = []

for s in seqs:

    X_tmp.append(s[:2])

    y.append(s[-1]-1)

y = np.array(y)





X = []

for val in X_tmp:

    package = []

    for i in range(len(val[0])):

        package.append([val[0][i], val[1][i]])

    X.append(package)

X = np.array(X)



X_train = X[:-int(X.shape[0]*0.2)]

X_test = X[-int(X.shape[0]*0.2):]

y_train = y[:-int(y.shape[0]*0.2)]

y_test = y[-int(y.shape[0]*0.2):]
sns.distplot(y)

class Model:

    def __init__(self, input_shape, batch_size, epochs):

        self.input_shape = input_shape

        self.batch_size = batch_size

        self.epochs = epochs

        

    def build(self):

        model = Sequential()

        model.add(CuDNNLSTM(128, input_shape=(self.input_shape), return_sequences=True))

        model.add(Dropout(0.2))

        model.add(BatchNormalization()) 



        model.add(CuDNNLSTM(128, return_sequences=True))

        model.add(Dropout(0.1))

        model.add(BatchNormalization())

        

        model.add(CuDNNLSTM(128))

        model.add(Dropout(0.2))

        model.add(BatchNormalization())

        

        model.add(Dropout(0.2))

        model.add(BatchNormalization())



        model.add(Dense(32, activation='relu'))

        model.add(Dropout(0.2))



        model.add(Dense(5, activation='softmax'))

        

        model.compile(

            loss='sparse_categorical_crossentropy',

            optimizer=Adam(lr=3e-5, decay=4e-8),

            metrics=['accuracy'],

        )

        self.model = model

        

    def train(self, X_train, y_train, X_test, y_test, visualize=False):

        self.history = self.model.fit(

            X_train, y_train,

            batch_size=self.batch_size,

            epochs=self.epochs,

            validation_data=(X_test, y_test),

            verbose=1,

        )

        

        if visualize:

            fig, axarr = plt.subplots(1, 2, figsize=(16, 8))

            axarr[0].plot(self.history.history['acc'])

            axarr[0].plot(self.history.history['val_acc'])

            axarr[0].set_title('Model accuracy')

            axarr[0].set_ylabel('Accuracy')

            axarr[0].set_xlabel('Epoch')

            axarr[0].legend(['Train', 'Test'], loc='upper left')



            axarr[1].plot(self.history.history['loss'])

            axarr[1].plot(self.history.history['val_loss'])

            axarr[1].set_title('Model loss')

            axarr[1].set_ylabel('Loss')

            axarr[1].set_xlabel('Epoch')

            axarr[1].legend(['Train', 'Test'], loc='upper left')

            

            plt.show()
model = Model(input_shape=X_train.shape[1:], batch_size=32, epochs=15)

model.build()

model.train(X_train, y_train, X_test, y_test, visualize=True)
model = Model(input_shape=X_train.shape[1:], batch_size=16, epochs=20)

model.build()

model.train(X_train, y_train, X_test, y_test, visualize=True)
model = Model(input_shape=X_train.shape[1:], batch_size=64, epochs=15)

model.build()

model.train(X_train, y_train, X_test, y_test, visualize=True)
model = Model(input_shape=X_train.shape[1:], batch_size=64, epochs=20)

model.build()

model.train(X_train, y_train, X_test, y_test, visualize=True)
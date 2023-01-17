import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense,Input,LeakyReLU

from keras.models import Model,load_model

from keras.optimizers import Adam,Adamax,RMSprop

from keras.utils import Sequence

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import seaborn as sns

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
BATCH_SIZE=256

EPOCHS=1000

LR=0.001
print(df.shape)

df.head(10)
column_names=df.columns

column_names=column_names.drop('Time')
df_zero=df[df['Class']==0]

df_one=df[df['Class']==1]
df_test=df_zero.iloc[:100,]

df_zero=df_zero.iloc[100:,] #remove those samples from training 

df_test=pd.concat([df_test,df_one])
train_df,val_df=train_test_split(df_zero[column_names],test_size=0.2)

print('Size of train data: {}'.format(train_df.shape))

print('Size of val data: {}'.format(val_df.shape))
train_df=train_df.drop('Class',axis=1)

val_df=val_df.drop('Class',axis=1)
minmax=MinMaxScaler()

train_df=minmax.fit_transform(train_df)

val_df=minmax.transform(val_df)
class Generator(Sequence):

    def __init__(self,dataframe,batch_size):

        self.dataframe=dataframe

        self.batch_size=batch_size

        self.shuffle=True

        self.on_epoch_end()

    

    def __len__(self):

        return int(len(self.dataframe)//self.batch_size)

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.dataframe))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __getitem__(self,index):

        indexes=self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.dataframe[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)



        return X, y

    

    def __data_generation(self, list_IDs_temp):

        return np.array(list_IDs_temp),np.array(list_IDs_temp)
train_gen=Generator(train_df,batch_size=BATCH_SIZE)

val_gen=Generator(val_df,batch_size=BATCH_SIZE)
inp=Input(shape=(29,))

x=Dense(16)(inp)

x=LeakyReLU()(x)

x=Dense(8)(x)

x=LeakyReLU()(x)

x=Dense(16)(x)

x=LeakyReLU()(x)

out=Dense(29,activation='sigmoid')(x)

model=Model(inp,out)

model.summary()
mc=ModelCheckpoint('autoenc.h5',period=1,save_best_only=True)

rop=ReduceLROnPlateau(monitor='val_loss',mode='min',period=5,min_lr=0.0000001)

es=EarlyStopping(monitor='val_loss',mode='min',patience=10)



model.compile(loss='binary_crossentropy',optimizer=Adam(LR))
result=model.fit_generator(train_gen,steps_per_epoch=train_gen.__len__(),epochs=EPOCHS,validation_data=val_gen,

                   validation_steps=val_gen.__len__(),callbacks=[mc,rop,es])
loss = result.history['loss']

val_loss = result.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'b', color='red', label='Training loss')

plt.plot(epochs, val_loss, 'b',color='blue', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
df_test=shuffle(df_test)

df_test_y=df_test['Class']

df_test=df_test[column_names].drop('Class',axis=1)

df_test=minmax.transform(df_test)

model=load_model('autoenc.h5')

print('Model Loaded')
preds=[]

for i in range(len(df_test)):

    sample=np.array(df_test[i])

    sample=np.expand_dims(sample,0)

    t_loss=model.evaluate(sample,sample,verbose=0)

    if t_loss>np.max(loss):

        preds.append(1)

    else:

        preds.append(0)
print('Accuracy: \n',accuracy_score(list(df_test_y),preds))

print('F1 score: \n',f1_score(list(df_test_y),preds))

print('Roc_Auc Score: \n',roc_auc_score(list(df_test_y),preds))

sns.heatmap(confusion_matrix(list(df_test_y),preds),annot=True)
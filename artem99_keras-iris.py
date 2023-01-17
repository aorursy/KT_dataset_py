



import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical

from keras.models import Model

from sklearn import preprocessing

from keras import Input

from keras.layers import Dense,concatenate,Dropout

path='/kaggle/input/iris/Iris.csv'
data=pd.read_csv(path,sep=',')

data.drop("Id", axis = 1, inplace = True)

data.head()
data.plot(subplots=True, figsize=(12, 10), grid=False,title="")#ГРАФИКИ КАЖДЫХ ПРИЗНАКОВ

plt.show()
plt.figure(figsize=(30,20))

sns.heatmap(data.corr(),cmap='coolwarm',annot = True)

plt.show()
datax=data.iloc[:,[0,1,2,3,4]].values

np.random.shuffle(datax)

datay=datax[:,4]

datax=datax[:,:-1]
print(datay.shape)

print(datax.shape)
scal=StandardScaler()

scal.fit(datax)

datax=scal.transform(datax)
SepalLengthCm=datax[:,0]

SepalWidthCm=datax[:,1]

PetalLengthCm=datax[:,2]

PetalWidthCm=datax[:,3] 
SepalLengthCm=SepalLengthCm.reshape(150,1)

SepalWidthCm=SepalWidthCm.reshape(150,1)

PetalLengthCm=PetalLengthCm.reshape(150,1)

PetalWidthCm=PetalWidthCm.reshape(150,1)
prep=preprocessing.LabelEncoder()

prep.fit(datay)

Species=prep.transform(datay)

Species
Species=to_categorical(Species)
print(Species.shape)
input_SepalLengthCm=Input(shape=(SepalLengthCm.shape[1],),dtype='float32')

x_1=Dense(64,activation='elu')(input_SepalLengthCm)

x_1=Dropout(0.5)(x_1)





input_SepalWidthCm=Input(shape=(SepalWidthCm.shape[1],),dtype='float32')

x_2=Dense(64,activation='elu')(input_SepalWidthCm)

x_2=Dropout(0.5)(x_2)



input_PetalLengthCm=Input(shape=(PetalLengthCm.shape[1],),dtype='float32')

x_3=Dense(64,activation='elu')(input_PetalLengthCm)

x_3=Dropout(0.5)(x_3)



input_PetalWidthCm=Input(shape=(PetalWidthCm.shape[1],),dtype='float32')

x_4=Dense(64,activation='elu')(input_PetalWidthCm)

x_4=Dropout(0.5)(x_4)



concatenated=concatenate([x_1,x_2,x_3,x_4])



answer=Dense(3,activation='softmax')(concatenated)

model=Model([input_SepalLengthCm,input_SepalWidthCm,input_PetalLengthCm,input_PetalWidthCm],answer)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
history=model.fit([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm],Species,epochs=20, batch_size=1,validation_split=0.3)
def result_eva (loss,val_loss,acc,val_acc):

    import matplotlib.pyplot as plt

    %matplotlib inline

    

    epochs = range(1,len(loss)+1)

    plt.plot(epochs, loss,'b-o', label ='Training Loss')

    plt.plot(epochs, val_loss,'r-o', label ='Validation Loss')

    plt.title("Training and Validation Loss")

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()

    

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, "b-o", label="Training Acc")

    plt.plot(epochs, val_acc, "r-o", label="Validation Acc")

    plt.title("Training and Validation Accuracy")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.show()
result_eva(history.history['loss'], history.history['val_loss'], history.history['acc'], history.history['val_acc'])
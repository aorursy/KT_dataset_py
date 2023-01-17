# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import cv2 as cv

from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D
from keras import models
from keras.optimizers import Adam,RMSprop 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization

import pickle


train = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
print(train.shape)


test= pd.read_csv('/kaggle/input/mnist-in-csv/mnist_test.csv')
print(test.shape)

np.random.seed(1)
amestec_train = train.iloc[np.random.permutation(len(train))] #se amesteca datele din train
print(amestec_train.shape)
amestec_train.head(5)
sample_size = amestec_train.shape[0] #numarul de valori din train
validation_size = int(amestec_train.shape[0]*0.1) #numarul de valori din setul de validare (10% din setul de antrenare)

#rearanjarea in format 54000, 28, 28, 1 si 54000, 1 pentru valorile de antrenare si validare
train_x = np.asarray(amestec_train.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1])
train_y = np.asarray(amestec_train.iloc[:sample_size-validation_size,0]).reshape([sample_size-validation_size,1])

val_x = np.asarray(amestec_train.iloc[sample_size-validation_size:,1:]).reshape([validation_size,28,28,1])
val_y = np.asarray(amestec_train.iloc[sample_size-validation_size:,0]).reshape([validation_size,1])

train_x.shape,train_y.shape, val_x.shape, val_y.shape

#rearanjarea testului
test_x = np.asarray(test.iloc[:,1:]).reshape([test.shape[0], 28, 28,1])
test_y = np.asarray(test.iloc[:,0]).reshape([test.shape[0],1])
test_y=test_y.astype('int32')
test_x.shape, test_y.shape
#normalizarea intre 0 si 1
train_x = train_x/255
val_x = val_x/255
test_x = test_x/255
#afisarea graficului cu rata de aparitie a cifrelor in setul de antrenare
counts = amestec_train.iloc[:sample_size-validation_size,:].groupby('label')['label'].count()

f = plt.figure(figsize=(10,6))
f.add_subplot(111)

plt.bar(counts.index,counts.values,width = 0.8,color="red")
for i in counts.index:
    plt.text(i,counts.values[i]+50,str(counts.values[i]),horizontalalignment='center',fontsize=14)

plt.tick_params(labelsize = 14)
plt.xticks(counts.index)
plt.xlabel("Cifre",fontsize=16)
plt.ylabel("Rata de aparitie",fontsize=16)
plt.title("Rata de aparitie a cifrelor in setul de antrenare",fontsize=20)
plt.show()
counts2 = amestec_train.iloc[sample_size-validation_size:,:].groupby('label')['label'].count()

f = plt.figure(figsize=(10,6))
f.add_subplot(111)

plt.bar(counts2.index,counts2.values,width = 0.8,color="red")
for i in counts2.index:
    plt.text(i,counts2.values[i]+5,str(counts2.values[i]),horizontalalignment='center',fontsize=14)

plt.tick_params(labelsize = 14)
plt.xticks(counts2.index)
plt.xlabel("Cifre",fontsize=16)
plt.ylabel("Rata de aparitie",fontsize=16)
plt.title("Rata de aparitie a cifrelor in setul de validare",fontsize=20)
plt.show()
#afisarea unor exemple de cifre din baza de date
rows = 5 # numarul de linii
cols = 6 # numarul de coloane

f = plt.figure(figsize=(2*cols,2*rows)) # se defineste figura

for i in range(rows*cols): 
    f.add_subplot(rows,cols,i+1) # se adauga un subplot la fiecare iteratie
    plt.imshow(train_x[i].reshape([28,28]),cmap="Blues") 
    plt.axis("off")
    plt.title(str(train_y[i]), y=-0.15,color="green")
#definirea modelului de arhitectura CNN

model = models.Sequential()

# Block 1
model.add(Conv2D(96,11, padding  ="same",input_shape=(28,28,1)))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Conv2D(256,5, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.25))


model.add(Conv2D(384,3, padding  ="same"))
model.add(LeakyReLU())

model.add(Conv2D(384,3, padding  ="same"))
model.add(LeakyReLU())

model.add(Conv2D(256,3, padding  ="same"))
model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(4096,activation='relu'))
model.add(Dense(4096,activation='relu'))
model.add(Dense(10,activation="sigmoid"))

model.summary()
#compilarea modelului
initial_lr = 0.001
loss = "sparse_categorical_crossentropy"
model.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])
model.summary()
#antrenarea
epochs = 20
batch_size = 256
history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=[val_x,val_y])
f = plt.figure(figsize=(20,7))

#Adding Subplot 1 (For Accuracy)
f.add_subplot(121)

plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set
plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

plt.title("Curba acuratetii",fontsize=18)
plt.xlabel("Epoci",fontsize=15)
plt.ylabel("Acuratete",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

#Adding Subplot 1 (For Loss)
f.add_subplot(122)

plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set

plt.title("Curba pierderii",fontsize=18)
plt.xlabel("Epoci",fontsize=15)
plt.ylabel("Pierdere",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()
#matricea de confuzie pentru validare
val_p = np.argmax(model.predict(val_x),axis =1)

error = 0
confusion_matrix = np.zeros([10,10])
for i in range(val_x.shape[0]):
    confusion_matrix[val_y[i],val_p[i]] += 1
    if val_y[i]!=val_p[i]:
        error +=1
        
print("Matricea de confuzie: \n\n" ,confusion_matrix)
print("\nErori in setul de validare: " ,error)
print("\nRata de eroare : " ,(error*100)/val_p.shape[0])
print("\nAcuratete : " ,100-(error*100)/val_p.shape[0])
print("\nDimensiunea setului de validare :",val_p.shape[0])
f = plt.figure(figsize=(10,8.5))
f.add_subplot(111)

plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")
plt.tick_params(size=5,color="white")
plt.xticks(np.arange(0,10),np.arange(0,10))
plt.yticks(np.arange(0,10),np.arange(0,10))

threshold = confusion_matrix.max()/2 

for i in range(10):
    for j in range(10):
        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")
        
plt.xlabel("Output")
plt.ylabel("Target")
plt.title("Matrice de confuzie a setului de validare")
rows = 4
cols = 9

f = plt.figure(figsize=(2*cols,2*rows))
sub_plot = 1
for i in range(val_x.shape[0]):
    if val_y[i]!=val_p[i]:
        f.add_subplot(rows,cols,sub_plot) 
        sub_plot+=1
        plt.imshow(val_x[i].reshape([28,28]),cmap="Blues")
        plt.axis("off")
        plt.title("T: "+str(val_y[i])+" P:"+str(val_p[i]), y=-0.15,color="Red")
plt.show()
test_y2 = np.argmax(model.predict(test_x),axis =1)

error2 = 0
confusion_matrix = np.zeros([10,10])
for i in range(test_x.shape[0]):
    confusion_matrix[test_y[i],test_y2[i]] += 1
    if test_y[i]!=test_y2[i]:
        error2 +=1
        

print("Matricea de confuzie: \n\n" ,confusion_matrix)
print("\nErori in setul de validare: " ,error2)
print("\nRata de eroare : " ,(error2*100)/test_y2.shape[0])
print("\nAcuratete : " ,100-(error2*100)/test_y2.shape[0])
print("\nDimensiunea setului de testare :",test_y2.shape[0])
f = plt.figure(figsize=(10,8.5))
f.add_subplot(111)

plt.imshow(np.log2(confusion_matrix+1),cmap="Reds")
plt.tick_params(size=5,color="white")
plt.xticks(np.arange(0,10),np.arange(0,10))
plt.yticks(np.arange(0,10),np.arange(0,10))

threshold = confusion_matrix.max()/2 

for i in range(10):
    for j in range(10):
        plt.text(j,i,int(confusion_matrix[i,j]),horizontalalignment="center",color="white" if confusion_matrix[i, j] > threshold else "black")
        
plt.xlabel("Output")
plt.ylabel("Target")
plt.title("Matrice de confuzie a setului de testare")
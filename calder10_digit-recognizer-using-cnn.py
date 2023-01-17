# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from keras.utils import to_categorical

import os

import time

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,BatchNormalization

from keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import seaborn as sns

from keras import optimizers

from sklearn.metrics import classification_report
# Caricamento e pre-processing dei dati

path_train='../input/train.csv'

path_test="../input/test.csv"



# Carico all'interno di un dataframe train e test set

train=pd.read_csv(path_train)

test=pd.read_csv(path_test)



# Creo gli array numpy per le features e i target sia per il train che per il test set

X_train=train.drop("label",axis=1).values

Y_train=train["label"].values

X_test=test.values



# Avendo a che fare con immagini è buona norma normalizzare i dati

X_train=X_train/X_train.max()

X_test=X_test/X_test.max()
# Visulizziamo le prima occorrenza di ogni immagine nel train set

plt.figure(figsize = (10,10))

for i in range(0,10):

    plt.subplot(3,4,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    m=X_train[Y_train==i][0].reshape([28,28])

    plt.xlabel(str(i))

    plt.imshow(m,cmap="gray")

plt.suptitle("First occurrence of each digit in train set",fontsize=18)

plt.savefig("Firs_occ_digits")

# Creiamo delle variabili dummy per i label

label=[0,1,2,3,4,5,6,7,8,9]

nc=10

Y_train_d=to_categorical(Y_train,10)



# Effettuo il resize dell'immagine ed aggiungo un'ulteriore dimensione contenete il numero di canali dell'immagine

X_train_c=X_train.reshape(-1,28,28,1)

X_test_c=X_test.reshape(-1,28,28,1)
np.random.seed(2)

m=Sequential()

m.add(Conv2D(filters=128,kernel_size=4,padding="same",activation="relu",input_shape=(28,28,1)))

m.add(Conv2D(filters=128,kernel_size=4,padding="same",activation="relu"))

m.add(MaxPooling2D(pool_size=2,strides=2))

m.add(Dropout(0.2))

m.add(Conv2D(filters=64,kernel_size=4,padding="same",activation="relu",))

m.add(Conv2D(filters=64,kernel_size=4,padding="same",activation="relu"))

m.add(MaxPooling2D(pool_size=2,strides=2))

m.add(Dropout(0.2))

m.add(Flatten())

m.add(Dense(1024,activation="relu"))

m.add(Dropout(0.2))

m.add(Dense(512,activation="relu"))

m.add(Dropout(0.4))

m.add(Dense(256,activation="relu"))

m.add(Dropout(0.6))

m.add(Dense(128,activation="relu"))

m.add(Dense(nc,activation='softmax'))

m.summary()
# Utilizziamo adesso come ottimizzatore l'adam

# Come funzione i costo la : categorical crossentropy

# Utilizziamo l'early stopping in modo che se dopo un certo numero di epcohe non si migliora di un certo min_delta

# l'Addestramento viene bloccato

el=EarlyStopping(monitor='val_loss',min_delta=0.001,patience=5,restore_best_weights=True)

ad=optimizers.Adam(lr=0.002,beta_1=0.9,beta_2=0.999,decay=0.004)

m.compile(loss="categorical_crossentropy",optimizer=ad,metrics=["accuracy"])

s=time.time()

h=m.fit(X_train_c,Y_train_d,batch_size=32,validation_split=0.4,epochs=50,callbacks=[el])

e=time.time()

t=e-s

print("Addestramento completato in %d minuti e %d secondi" %(t/60,t*60))

acc=h.history['acc']

val_acc=h.history['val_acc']

loss=h.history['loss']

val_loss=h.history['val_loss']
# Trend of accuracy during the training 

plt.plot(acc)

plt.plot(val_acc)

plt.title('Digit Recognizer Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train','Validation'])

plt.show()
# Trend of loss during the training 

plt.plot(loss)

plt.plot(val_loss)

plt.title('Digit Recognizer Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train','Validation'])

plt.show()


y_pred_train=m.predict(X_train_c)

y_pred_train = np.argmax(y_pred_train,axis = 1)



y_pred_train=np.array(y_pred_train,dtype='int')

Y_train=np.array(Y_train,dtype='int')



cm=confusion_matrix(Y_train,y_pred_train)

cm_df = pd.DataFrame(cm,index = ['0','1','2','3','4','5','6','7','8','9'],  columns =['0','1','2','3','4','5','6','7','8','9'])

plt.figure(figsize=(10,10))

sns.heatmap(cm_df,annot=True,cmap="Blues_r",linewidth=0.5,square=True,fmt='g')



plt.ylabel("True Label ")

plt.xlabel("Predict Label")

plt.title("CONFUSION MATRIX FOR TRAINING SET")



a=sum(np.diag(cm))

b=sum(sum(cm))

acc=(a/b)*100

print('Accuracy on Training Set: %.2f ' %(acc))



y_test=m.predict(X_test_c)

y_test = np.argmax(y_test,axis = 1)

out=pd.DataFrame({"ImageId": list(range(1,len(y_test)+1)),"Label": y_test})

out.to_csv("Submission_cnn.csv", index=False, header=True)
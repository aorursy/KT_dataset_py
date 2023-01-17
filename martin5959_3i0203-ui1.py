# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import python knižníc
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline
# Načítanie dát train.csv a test.csv
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print('Tvar súboru tréningových dát: ', train_data.shape)
print('Tvar súboru testovacích dát: ', test_data.shape)
# Zobrazenie dát a vykreslenie
XT = train_data.drop(labels = ['label'], axis= 1)
YT = train_data['label']

# Vymazanie voľného priestoru
del train_data

# Vykreslenie počtu jednotlivých číslic
sns.countplot(YT)
# Údaje sú momentálne vo formáte int8, takže predtým, ako sa vložia do neurónovej siete sa musí previesť typ údajov na float32
XT = XT.astype('float32')
test = test_data.astype('float32')
XT = XT/255.
test = test/255.
# Zmena tvaru obrázka v 3 rozmeroch (výška = 28 pixelov, šírka = 28 pixelov, kanál = 1)
XT = XT.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
XT.shape, test.shape
# Zmena kategorických označení pomocou one-hot encoding (proces, ktorým sa kategorické premenné prevádzajú do formy, 
# ktorá by sa mohla poskytnúť ML algoritmom, aby mohli robiť lepšiu prácu v predikcii)
YT_OHE = to_categorical(YT)

# Zobrazenie zmeny kategorických označení pomocou one-hot encoding
print('Pôvodné označenie:', YT[0])
print('Po konverzii pomocou one-hot encoding:', YT_OHE[0])
# Rozdelenie súboru údajov na tréningové a testovacie súbory údajov pomocou nástroja knižnice SciKit Learn
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(XT, YT_OHE, test_size=0.2, random_state=21)
# Import open-source knižnice neurónovej siete s názvom Keras
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
# Množstvo údajov, ktoré vidí každá iterácia v epoche
batch_size = 128
# Počet epoch trénovania modelu
epochs = 20
num_classes = 10
# Implementácia KERAS CNN
model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3),padding = 'same', 
                 activation ='relu', input_shape = (28,28,1)))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D((2, 2),padding='same'))

model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (3,3),padding = 'same', 
                 activation ='relu'))

model.add(LeakyReLU(alpha=0.1))

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size = (3,3),padding = 'same', 
                 activation ='relu'))

model.add(LeakyReLU(alpha=0.1))                  

model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(LeakyReLU(alpha=0.1))           

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))
# Kompilácia modelu
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
# Zhrnutie modelu
model.summary()
train_model = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
# Vykreslenie grafov presnosti a chyby predikcie tréningu a validácie

accuracy = train_model.history['accuracy']

val_accuracy = train_model.history['val_accuracy']

loss = train_model.history['loss']

val_loss = train_model.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Presnosť tréningu')

plt.plot(epochs, val_accuracy, 'b', label='Presnosť validácie')

plt.title('Presnosť výcviku a validácie')

plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Chyba predikcie tréningu')

plt.plot(epochs, val_loss, 'b', label='Chyba predikcie validácie')

plt.title('Chyba predikcie tréningu a validácie')

plt.legend()
plt.show()
hodnotenie = model.evaluate(valid_X, valid_label, verbose=1)
# Výpis presnosti a chyby predikcie validačného testu
print('Chyba predikcie  validačného testu:', hodnotenie[0])
print('Presnosť validačného testu:', hodnotenie[1])
# Predikcia výsledkov
predict_classes = model.predict(test)

# Výber indexu s maximálnou pravdepodobnosťou
predict_classes = np.argmax(np.round(predict_classes),axis=1)

results = pd.Series(predict_classes,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)
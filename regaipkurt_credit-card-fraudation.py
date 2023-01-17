import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
veri = pd.read_csv("../input/creditcard.csv")
print("Veri Setimiz:\n {} \nVerinin uzunluğu: {} ".format(veri.head(),len(veri)))

pozitifler = veri[veri["Class"]==1] # pozitif örneklerin olduğu durumları aldık sadece

negatifler = veri[veri["Class"]==0] # sadece negatif örneklerin olduğu durumlar

print("\n\nPOZİTİF ÖRNEKLER:\n  {}\nPozitif veri olan durumların sayısı: {}".format(pozitifler.head(), len(pozitifler)))

print("\n\nNEGATİF ÖRNEKLER:\n {}\nNegatif veri olan durumların sayısı: {}".format(negatifler.head(), len(negatifler)))
negatif_veri = negatifler.iloc[:19508,:] #ilk 19508 satırı aldım. (Tam 20000 verimiz olsun diye :D)

print(len(negatif_veri))

print(negatif_veri.head())



temiz_veri = pd.concat([negatif_veri, pozitifler], axis=0, ) #pozitif ve negatifleri birleştiriyoruz.

print(temiz_veri)

from sklearn.utils import shuffle

temiz_veri = shuffle(temiz_veri) #shuffle ile verimizin sırasını karıştırıyoruz ki 1'ler son tarafa toplanmasın

print(temiz_veri)
temiz_veri = temiz_veri.drop(columns="Time") # Time kolonunu çıkarıyoruz.

print(temiz_veri.head())
kesif_degerleri = temiz_veri.describe()

print(kesif_degerleri)



plt.plot(temiz_veri.std()) # standart sapmaların grafiği

print("\nAmount Kolonu Standart Sapması: ", temiz_veri["Amount"].std())
print("\nVerinin özellikleri:\n",temiz_veri.describe())



x = temiz_veri.iloc[:,:29].values

y = temiz_veri.iloc[:,29:].values



print("\nX'in Boyutu: {}\nY'nin Boyutu: {}\n".format(x.shape, y.shape))



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.33)



print("\nx_train'in boyutu: {}\ny_train'in boyutu: {}\n".format(x_train.shape, y_train.shape))

print("\nx_test'in boyutu: {}\ny_test'in boyutu: {}\n".format(x_test.shape, y_test.shape))

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train, x_test = ss.fit_transform(x_train), ss.fit_transform(x_test)

print(temiz_veri)
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.optimizers import SGD



model = Sequential()

model.add(Dense(64, kernel_initializer="glorot_uniform", activation="relu"))

model.add(Dense(64, activation="tanh"))

model.add(Dense(64, activation="relu"))

model.add(Dense(64, activation="tanh"))

model.add(Dense(1))



sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss="mse", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=1000)
y_pred = model.predict(x_test)

a = 0

for i in y_pred:

    if i > 0.5:

        y_pred[a] = 1

    else:

        y_pred[a] = 0

    a += 1

print(y_test.shape, y_pred.shape)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)

print(cm)

print(np.count_nonzero(y_pred))

print(np.count_nonzero(y_test))

toplam = cm.sum()

hatalılar = cm[0,1] + cm[1,0]

hata_oranı = (hatalılar/toplam) * 100

print("Hata oranı: %{}".format(hata_oranı))
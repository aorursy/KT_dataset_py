

import numpy as np # arrayleri işlemek için

import pandas as pd #heart.csv datasını işlemek için



import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #input içinde bulunan dosyaları yazdır



data = pd.read_csv("../input/heart.csv")# Heart.csv verimizi data ya çekiyoruz

data.info() # Veri hakkinda bilgi alıyoruz

data.head() # sütünları ve satırları inceliyoruz

y_data = data.target.values # Verideki  Target Sütununu y_data ya eşitliyoruz 

x_data = data.drop(['target'], axis=1) # Geri kalan tüm Sütunlar x_data ya eşitleniyor

 
# %% normalization

#Normalizasyon (Ayrıştırma), veritabanlarında çok fazla sütun ve satırdan oluşan bir tabloyu tekrarlardan arındırmak için daha az satır ve sütun içeren alt kümelerine ayrıştırma işlemidir.

X = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

Y = y_data.reshape(X.shape[0],1) # Y mizi 303 e 1 lik matrix haline getiriyoruz 

print(Y)
#matrizlerimizin boyutlarına bakıyoruz

print("X shape: " , X.shape)

print("Y shape: " , Y.shape)
from keras.models import Sequential

from keras.layers import Dense

model = Sequential()

model.add(Dense(12, input_dim=13, init='uniform', activation='relu'))

model.add(Dense(8, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X,Y,epochs=50,batch_size=10,validation_split=0.13)

predictions = model.predict(X)
predict = numpy.array([44,1,0,112,290,0,0,153,0,0,2,1,2]).reshape(1,13)

print(model.predict_classes(predict))
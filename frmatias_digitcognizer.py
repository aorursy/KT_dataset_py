!pip install keras


#libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from keras.utils import np_utils

from sklearn.model_selection import StratifiedKFold



#loadind data

previsores = pd.read_csv('../input/train.csv')

classe = pd.read_csv('../input/train.csv')      

#extracting labels

classe = classe.iloc[: , 0:1]

#deliting labels from predict data

previsores = previsores.drop('label', axis = 1)

#reducing values [0~1]

previsores = previsores.astype('float32')

previsores /= 255

previsores = previsores.values.reshape(-1, 28, 28, 1)

classe = np_utils.to_categorical(classe, 10)

#Model

classificador = Sequential()

classificador.add(Conv2D(64, (3,3), input_shape=(28,28,1), activation = 'relu'))

classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dropout(0.1))

classificador.add(Dense(units = 64, activation = 'relu'))

classificador.add(Dense(units = 10, activation = 'softmax'))

classificador.compile(loss = 'categorical_crossentropy', optimizer = 'Adamax',metrics = ['accuracy'])

classificador.fit(previsores, classe,batch_size = 300, epochs = 500)



# Any results you write to the current directory are saved as output.
#prepering test data

teste = pd.read_csv('../input/test.csv')

teste /= 255



teste = teste.values.reshape(-1, 28, 28, 1)

# predict results

results = classificador.predict(teste)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("sub.csv",index=False)
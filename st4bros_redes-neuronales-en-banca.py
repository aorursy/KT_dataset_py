import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv('../input/Bank_registries.csv')

print(dataset.shape)

dataset.head()
X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values

pd.DataFrame(X[0:4])
#Label Encoder transforma a numeros los niveles de la variable categorica. 

#OneHotEncoder desdobla en k-columnas binarias los k-niveles de cada variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



#Cargamos el modelo y transformamos los niveles categoricos a numeros consecutivos para (Geograpy y gender)

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()

X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

pd.DataFrame(X).describe()

print(dataset.columns)

X[1:10]

#hacemos Dummy Encodign, generando k-1 nuevas columnas para los k niveles de las variables categoricas

onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

pd.DataFrame(X).describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import keras

from keras.models import Sequential

from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 500)

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

cm
good = (cm[0][0] + cm[1][1])/np.sum(cm)

print (good)
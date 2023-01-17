import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)





previsorestrain = train.iloc[:, 0:784].values

classetrain = train.iloc[:, 0].values



previsorestest = test.iloc[:, 0:784].values

classetest = test.iloc[:, 0].values



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

previsorestrain = scaler.fit_transform(previsorestrain)

previsorestest = scaler.fit_transform(previsorestest)



from sklearn.svm import SVC

classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0, gamma='auto')

classificador.fit(previsorestrain, classetrain)

previsoes = classificador.predict(previsorestest)



from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classetest, previsoes)

matriz = confusion_matrix(classetest, previsoes)



import collections

collections.Counter(classetest)

print(matriz)

print(previsoes)

type(previsoes)

type(classetest)

result = np.array([previsoes, classetest])

np.savetxt("result.csv", result, delimiter=",")

print(result)



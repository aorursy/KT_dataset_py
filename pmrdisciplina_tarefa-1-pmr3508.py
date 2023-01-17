import pandas as pd
import os
print (os.listdir('../input/adult-data'))
train = pd.read_csv('../input/adult-data/train_data.csv', na_values = '?')
test = pd.read_csv('../input/adult-data/test_data.csv', na_values = '?')
train.head()
#nao usei fnlwgt para aumentar acuracia
Xtrain = train[['education.num', 'capital.gain', 'capital.loss', 'hours.per.week']] 
Ytrain = train[['income']]

print(Xtrain.head())
print(Ytrain.head())
print(Xtrain.shape, Ytrain.shape)
print(Xtrain.dropna().shape, Ytrain.dropna().shape)
#percebe-se que não há dados NaN nas variáveis quantitativas
Xtest = test[['education.num', 'capital.gain', 'capital.loss', 'hours.per.week']]

print(Xtest.shape)
print(Xtest.dropna().shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, Xtrain, Ytrain.values.ravel(), cv = 5)
    
    print(i, np.mean(scores))
knn = KNeighborsClassifier(n_neighbors = 16)
knn.fit(Xtrain, Ytrain.values.ravel())
Ypred = knn.predict(Xtest) #predição type = numpy array
Ypred_d = pd.DataFrame(data = Ypred) #transformo a minha numpy array em um DataFrame de Pandas
Ypred_d.to_csv('submission7.csv') #crio o arquivo para submeter no Kaggle
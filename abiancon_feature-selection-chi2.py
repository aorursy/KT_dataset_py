# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import chi2

from sklearn.feature_selection import SelectKBest
train = pd.read_csv("../input/mobile-price-classification/train.csv")

array = train.values

X = array[:,0:20]

y = array[:,20]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)



scaler = StandardScaler()

scaler.fit(X)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=1, tol=1e-8, learning_rate_init=.01)

mlp.fit(X_train,y_train)



print('MLP senza feature_selection')

print('TrainAccuracy: ', mlp.score(X_train,y_train))

print('ValidationAccuracy: ', mlp.score(X_test, y_test))
#chi2_scores contiene i valori della chi2 mentre chi_2_p_value

#contiene i p-values. Le features a cui corrispondono p-values

#più grandi e chi2 più piccole possono essere scartate

chi2_scores, chi_2_p_value = chi2(X,y)

print('chi2 scores        ', chi2_scores)

print('chi2 p-value      ', chi_2_p_value)
lista = list(train)

lista.remove('price_range')



#SelectKBest permette di considerare le k features da selezionare per il training del modello

testchi2 = SelectKBest(score_func = chi2, k = 10)

fitchi2 = testchi2.fit(X, y) #Run score function on (X, y) and get the appropriate features.

dfscores = pd.DataFrame(fitchi2.scores_)

dfcolumns = pd.DataFrame(lista)

featureScores = pd.concat([dfcolumns, dfscores], axis = 1)

featureScores.columns = ['Features', 'Chi2-Score']

print('RISULTATO DI CHI2')

print(featureScores.nlargest(21,'Chi2-Score'))

print('\nFEATURES DA SELEZIONARE')

print(featureScores.nlargest(10,'Chi2-Score'))
X_trainf = testchi2.fit_transform(X,y)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_trainf, y, test_size=0.33, random_state=123)



scaler.fit(X_trainf)

X_train2 = scaler.transform(X_train2)

X_test2 = scaler.transform(X_test2)

mlp2 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, alpha=1, tol=1e-8, learning_rate_init=.01)

mlp2.fit(X_train2,y_train2)



print('MLP con feature_selection')

print('TrainAccuracy: ', mlp2.score(X_train2,y_train2))

print('ValidationAccuracy: ', mlp2.score(X_test2,y_test2))
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



train_data = pd.read_csv("../input/train.csv")

print('Cantidad de muestras de entrenamiento',len(train_data.index))

print('Cantidad de variables',len(train_data.columns)-2)
y = train_data['Mutacion'].values

X = train_data[train_data.columns[1:-1]].values

from sklearn.decomposition import PCA

n_components = 15

print("Extraemos los %d eigen principales de %d muestras" % (n_components, X.shape[0]))

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)

          
print("Proyectamos las muestras en la base ortonormal de valores eigen")          

X_pca = pca.transform(X)

from sklearn.svm import SVC

baseline_clf = SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)

  

baseline_clf = baseline_clf.fit(X_pca, y)

print(baseline_clf.score(X_pca,y))

#print(train_data['patientID'],y,baseline_clf.predict(X_pca))

print('Cargamos el set de test')

test_data = pd.read_csv("../input/test.csv")

print('Cantidad de muestras de entrenamiento',len(test_data.index))

print('Cantidad de variables',len(test_data.columns)-1)



X_test = test_data[test_data.columns[1:]].values



print('Aplicamos PCA al test')

X_test_pca = pca.transform(X_test)

print('Aplicamos el clasificador al test')

preds = baseline_clf.predict(X_test_pca)



for i,pred in enumerate(preds):

    print(test_data['patientID'][i],pred)

print('Cantidad de 1', list(preds).count(1))

print('Cantidad de 0', list(preds).count(0))



print("Creamos un archivo CSV con los resultados.")

submission_data = pd.DataFrame()

submission_data['patientID'] = test_data['patientID']

submission_data['Mutacion'] = preds

submission_data.to_csv('submission_pca_svm.csv',index=False)
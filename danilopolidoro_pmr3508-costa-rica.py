import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import threading
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

%config InlineBackend.figure_format = 'svg'
mpl.rcParams['figure.dpi']= 300
pd.set_option('display.expand_frame_repr', False)
costaRica = pd.read_csv('../input/train.csv').dropna()
costaRica.iloc[:,0:14].head()
costaRica.iloc[:,14:28].head()
costaRica.iloc[:,28:42].head()
costaRica.iloc[:,42:56].head()
costaRica.iloc[:,56:70].head()
costaRica.iloc[:,70:84].head()
costaRica.iloc[:,84:98].head()
costaRica.iloc[:,98:112].head()
costaRica.iloc[:,112:126].head()
costaRica.iloc[:,126:144].head()
newCosta = costaRica.drop(['Id','idhogar', 'dependency', 'edjefa', 'edjefe'], axis = 1)
accuracies = []
for i in range(100):
    classifier = KNeighborsClassifier(n_neighbors=i+1)
    scores = cross_val_score(classifier, newCosta.iloc[:,0:137],newCosta.iloc[:,137:138], cv = 10)
    accuracies.append(scores.mean())
    print('K = {0}; accuracy = {1}'.format(i+1, scores.mean()))
    
print('')
print('Best classifier at K = {0} with accuracy = {1}'.format(accuracies.index(max(accuracies))+1,max(accuracies)))
costaRicaTest = pd.read_csv('../input/test.csv')
costaRicaTestFill = costaRicaTest.fillna(0)
newCostaRicaTest = costaRicaTestFill.drop(['Id','idhogar', 'dependency', 'edjefa', 'edjefe'], axis = 1)
classifier = KNeighborsClassifier(n_neighbors=9)
classifier.fit(newCosta.iloc[:,0:137], newCosta.iloc[:,137:138])
prediction = classifier.predict(newCostaRicaTest)
import csv
ids = costaRicaTest.iloc[:,0:1].values.transpose()[0]
csvFile = open('submission.csv', mode = 'w')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['Id', 'Target'])
for index, element in enumerate(ids):
    csvWriter.writerow([element, prediction[index]])
csvFile.close()

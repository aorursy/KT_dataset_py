# Importação de bibliotecas

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importação da base Spam (de treino)
spam_train =  pd.read_csv('../input/spambase/train_data.csv')
# Dimensão da base
spam_train.shape
# Início da tabela
spam_train.head()
spam_train.describe()
spam_train["word_freq_order"].value_counts() 
spam_train["word_freq_free"].value_counts() 
# Importação da biblioteca de plotagem de gráficos
import matplotlib.pyplot as plt
spam_train["ham"].value_counts()
spam_train["ham"].value_counts().plot(kind="pie")
# Retirada da coluna "ham"
spam_train_X = spam_train.drop('ham', axis=1)
spam_train_Y = spam_train.ham
cross_validation = []
for i in range (1,51):
    knn = KNeighborsClassifier(n_neighbors=i)
    cross_validation.append(cross_val_score(knn,spam_train_X,spam_train_Y,cv=10).mean())
for i in range (0,25):
    print(2*i+1,":  ",cross_validation[2*i],"    ", 2*i+2,":  ",cross_validation[2*i+1])
NBayes = MultinomialNB()
NBayes.fit(spam_train_X,spam_train_Y)
# Importação da base Spam (de teste)
spam_test =  pd.read_csv("../input/spambase/test_features.csv")
results = NBayes.predict(spam_test)
prediction = pd.DataFrame(columns = ['Id','ham'])
prediction['ham'] = results
prediction['Id'] = spam_test.Id
prediction.to_csv("pred.csv",index=False)
prediction
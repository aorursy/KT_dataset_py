import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
train = pd.read_csv("../input/spamdata/train_data.csv", index_col = 'Id')
train.info()
train.head()
train.describe()
#Spam
spam = train[(train['ham'] == 0)]

#Ham
ham = train[(train['ham'] == 1)]

#Diferença média ocorrencia spam e ham
spamHamMean = (ham.mean() - spam.mean())[train.columns[:54]]

#Diferença média ocorrencia spam e ham para colunas restantes
spamHamMeanCapital = (ham.mean() - spam.mean())[train.columns[54:57]]
plt.figure(figsize=(20,5))
spamHamMean.plot(kind = 'bar')
spam.iloc[:54].mean(axis=0)
spamHamMeanCapital.plot(kind = 'bar')
spam.iloc[:,54:57].mean(axis=0)
Xtrain = train[train.columns[0:57]]
Xtrain.head()
Ytrain = train['ham']
Ytrain.head()
naiveBayes = GaussianNB()
scores = cross_val_score(naiveBayes, Xtrain,Ytrain, cv=10)
scores
test = pd.read_csv('../input/spamdata/test_features.csv', index_col = 'Id')
test.head()
naiveBayes.fit(Xtrain,Ytrain)
Ytest = naiveBayes.predict(test)
prediction = pd.DataFrame(index = test.index)
prediction['ham'] = Ytest
prediction.head()

prediction.to_csv('result.csv',index = True)
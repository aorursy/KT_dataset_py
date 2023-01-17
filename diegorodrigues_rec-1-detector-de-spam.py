import sklearn

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
spamTrain = pd.read_csv("../input/pmr-3508-tarefa-2/train_data.csv",

        engine='python')



spamTrain.head()
Corr=spamTrain.corr(method='pearson')["ham"]



Corr.sort_values(ascending=True)
Corrbase=spamTrain[["word_freq_your", 

                        "word_freq_000",

                        "char_freq_$",

                        "word_freq_remove",

                        "word_freq_you",

                        "word_freq_free",

                        "word_freq_business",

                        "capital_run_length_total",

                        "word_freq_order",

                        "word_freq_receive",

                        "word_freq_our",

                        'char_freq_!',

                        'word_freq_over',

                        'word_freq_credit',

                        'word_freq_money',

                        'capital_run_length_longest',

                        'word_freq_internet',

                        'word_freq_all',

                        'word_freq_addresses',

                        'word_freq_email',

                        'word_freq_650',

                        'word_freq_1999',

                        'word_freq_labs',

                        'word_freq_george',

                        'word_freq_hpl',

                        'word_freq_hp','ham']]



Corrbase.head()
Hambase = Corrbase[Corrbase.ham == True]

Spambase = Corrbase[Corrbase.ham == False]
x = np.array(range(len(Hambase["word_freq_your"])))

y = np.array(range(len(Spambase["word_freq_your"])))



plt.figure(figsize=(30, 7))

plt.subplot(131)

plt.plot( x, Hambase["word_freq_your"],  color='grey', label='your') 

plt.plot( x, Hambase["word_freq_000"],  color='blue', label = '000')  

plt.plot( x, Hambase["char_freq_$"],  color='yellow', label = '$')  





plt.legend()

plt.title("Influência das palavras em ser Ham")



plt.grid(True)

plt.xlabel("emails ham")

plt.ylabel("frequências de your, 000 e $ ")



plt.subplot(132)

plt.plot( y, Spambase["word_freq_your"],  color='grey', label='your') 

plt.plot( y, Spambase["word_freq_000"],  color='blue', label = '000')  

plt.plot( y, Spambase["char_freq_$"],  color='yellow', label = '$')  





plt.legend()

plt.title("Influência das palavras em ser Spam")



plt.grid(True)

plt.xlabel("emails spam")

plt.ylabel("frequências de your, 000 e $ ")



plt.show()

x = np.array(range(len(Hambase["word_freq_hp"])))

y = np.array(range(len(Spambase["word_freq_hp"])))



plt.figure(figsize=(30, 7))

plt.subplot(131)

plt.plot( x, Hambase["word_freq_george"],  color='grey', label = 'george') 

plt.plot( x, Hambase["word_freq_hp"],  color='yellow', label='hp') 

plt.plot( x, Hambase["word_freq_hpl"],  color='blue', label = 'hpl')  

 





plt.legend()

plt.title("Influência das palavras em ser Ham")



plt.grid(True)

plt.xlabel("emails ham")

plt.ylabel("frequência de hp, hpl e george")



plt.subplot(132)

plt.plot( y, Spambase["word_freq_george"],  color='red', label = 'george') 

plt.plot( y, Spambase["word_freq_hp"],  color='yellow', label='hp') 

plt.plot( y, Spambase["word_freq_hpl"],  color='blue', label = 'hpl')  

 





plt.legend()

plt.title("Influência das palavras em ser Spam")



plt.grid(True)

plt.xlabel("emails spam")

plt.ylabel("frequência de hp, hpl e george ")



plt.show()

X_CorrTrain=Corrbase.drop(columns=["ham"])

Y_CorrTrain=Corrbase["ham"]



X_CorrTrain.head()
X_CorrTrain.shape
X_baseTest1= pd.read_csv("../input/pmr-3508-tarefa-2/test_features.csv",

        engine='python')





X_baseTest=X_baseTest1[["word_freq_your", 

                        "word_freq_000",

                        "char_freq_$",

                        "word_freq_remove",

                        "word_freq_you",

                        "word_freq_free",

                        "word_freq_business",

                        "capital_run_length_total",

                        "word_freq_order",

                        "word_freq_receive",

                        "word_freq_our",

                        'char_freq_!',

                        'word_freq_over',

                        'word_freq_credit',

                        'word_freq_money',

                        'capital_run_length_longest',

                        'word_freq_internet',

                        'word_freq_all',

                        'word_freq_addresses',

                        'word_freq_email',

                        'word_freq_650',

                        'word_freq_1999',

                        'word_freq_labs',

                        'word_freq_george',

                        'word_freq_hpl',

                        'word_freq_hp']]







X_baseTest.head()
X_baseTest.shape
from sklearn.naive_bayes import GaussianNB

gauss = GaussianNB()

gauss.fit(X_CorrTrain,Y_CorrTrain)

resultado = cross_val_score(gauss, X_CorrTrain, Y_CorrTrain, cv=10)

#print(resultado)

print("Acurácia:", resultado.mean())

print("Desvio Padrão:", resultado.std())
from sklearn.naive_bayes import BernoulliNB

bern = BernoulliNB()

bern.fit(X_CorrTrain,Y_CorrTrain)

resultado = cross_val_score(bern, X_CorrTrain, Y_CorrTrain, cv=10)

#print(resultado)

print("Acurácia:", resultado.mean())

print("Desvio Padrão:", resultado.std())
from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors=16)

#knn.fit(X_CorrTrain,Y_CorrTrain)



maior = 0.0

for i in range(2,15):

    media = 0.0

    knn = KNeighborsClassifier(n_neighbors = i)

    scores = cross_val_score(knn, X_CorrTrain, Y_CorrTrain, cv = 10)

    media=np.mean(scores)

    if (media>maior):

        maior = media

        k=i



resultado = cross_val_score(KNeighborsClassifier(n_neighbors = k), X_CorrTrain, Y_CorrTrain, cv=10)

print(resultado)

print("Acurácia:", resultado.mean())

print("Desvio Padrão:", resultado.std())
# exportação dos dados



X_test = X_baseTest

Y_predict = gauss.predict(X_test)



output = pd.concat( [X_baseTest1['Id'], pd.Series(Y_predict, name='ham')], axis=1)

output.to_csv('./predOut.csv', index=False)

output
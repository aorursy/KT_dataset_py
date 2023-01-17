import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from IPython.display import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
%matplotlib inline 

treino = pd.read_csv("../input/treino/train_data.csv")

treino.head( n =20 )
treino_nospam = treino.query('ham == 1')
treino_spam = treino.query('ham == 0')
nospam_hashtag_mean = np.mean(treino_nospam['char_freq_#'])
spam_hashtag_mean = np.mean(treino_spam['char_freq_#'])
locations = [1, 2]
heights = [nospam_hashtag_mean, spam_hashtag_mean]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa média de caracter # por tipo de email')
plt.ylabel('Taxa média de caracter #')
plt.xlabel('Tipo de email')
nospam_cifrao_mean = np.mean(treino_nospam['char_freq_$'])
spam_cifrao_mean = np.mean(treino_spam['char_freq_$'])
locations = [1, 2]
heights = [nospam_cifrao_mean, spam_cifrao_mean]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa média de caracter $ por tipo de email')
plt.ylabel('Taxa média de caracter $')
plt.xlabel('Tipo de email')
nospam_re_mean = np.mean(treino_nospam['word_freq_re'])
spam_re_mean = np.mean(treino_spam['word_freq_re'])
locations = [1, 2]
heights = [nospam_re_mean, spam_re_mean]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa média da palavra re por tipo de email')
plt.ylabel('Taxa média da palavra re')
plt.xlabel('Tipo de email')
nospam_address_mean = np.mean(treino_nospam['word_freq_address'])
spam_address_mean = np.mean(treino_spam['word_freq_address'])
locations = [1, 2]
heights = [nospam_address_mean, spam_address_mean]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa média da palavra address por tipo de email')
plt.ylabel('Taxa média da palavra address')
plt.xlabel('Tipo de email')
nospam_colchete_mean = np.mean(treino_nospam['char_freq_['])
spam_colchete_mean = np.mean(treino_spam['char_freq_['])
locations = [1, 2]
heights = [nospam_colchete_mean, spam_colchete_mean]
labels = ["Ham", "Spam"]
plt.bar(locations, heights, tick_label=labels)
plt.title('Taxa média do caracter colchete por tipo de email')
plt.ylabel('Taxa média do caracter colchete')
plt.xlabel('Tipo de email')
features_train = treino.drop(columns=['ham'])
target_train = treino['ham']

gnb = GaussianNB()

gnb.fit(features_train, target_train)
scores = cross_val_score(gnb, features_train, target_train, cv=10)

print(scores)
teste = pd.read_csv("../input/testespam/test_features.csv")
teste.head()

sample = pd.read_csv("../input/sample/sample_submission_1.csv")
sample.shape
sample.head()
predictions = gnb.predict(teste)
str(predictions)
ids = teste['Id']
# print(ids)
# arr1= teste.iloc[:,0].values
# arr1 = arr1.ravel()
entrega = pd.DataFrame({'Id':ids,'ham':predictions[:]})
entrega.to_csv("predictions.csv", index = False)
entrega.shape
entrega.head()

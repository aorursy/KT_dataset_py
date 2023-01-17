# Import libraries
import pandas as pd
import sklearn
import numpy as np
# Training database reading
trainDB = pd.read_csv("../input/dataset/train_data.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="NaN")
# Testing database reading
testDB = pd.read_csv("../input/dataset/test_features.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="NaN")
# Visualização da Base de Treino
trainDB.iloc[0:20,:]
# Tamanho base de treino
size_train = trainDB.shape
size_train
#import biblioteca de plot
import matplotlib.pyplot as plt
# Palavras que mais aparecem em spam
spam_data = trainDB[trainDB.ham == False]
somas = []
for i in range(1,49): # o número da última coluna com 'word_freq' é a 48
    # Verificando quais palavras acontecem mais vezes para a label spam 
    somas.append(np.count_nonzero(spam_data.iloc[:,i]))
var_names = list(trainDB)
spam_words = pd.DataFrame({'Words':var_names[0:48],'Times_app':somas})
spam_words
# Chars que mais aparecem em spam
somas_char = []
char_data = spam_data.iloc[:,48:54] # intervalo dos chars 
for i in range(6):
    # Verificando quais chars acontecem mais vezes para a label spam 
    somas_char.append(np.count_nonzero(char_data.iloc[:,i]))
spam_chars = pd.DataFrame({'Char':var_names[48:54],'Times_app':somas_char})
spam_chars
# Análise do 'capital_run_length_average'
crla = trainDB['capital_run_length_average']
clas = np.array([0,0,0,0])
for i in range(len(crla)):
    if crla[i] <= 1:
        clas[0] = clas[0] + 1
    if crla[i] > 1 and crla[i] <= 10:
        clas[1] = clas[1] + 1
    if crla[i] > 10 and crla[i] <= 100:
        clas[2] = clas[2] + 1
    if crla[i] > 100:
        clas[3] = clas[3] + 1
lab = '<1','1<X<10','10<X<100','>100'
plt.pie(clas,labels = lab);
# Análise do 'capital_run_length_total'
crlt = trainDB['capital_run_length_total']
clas2 = np.array([0,0,0,0])
for i in range(len(crlt)):
    if crla[i] <= 1:
        clas2[0] = clas2[0] + 1
    if crla[i] > 1 and crla[i] <= 100:
        clas2[1] = clas2[1] + 1
    if crla[i] > 100 and crla[i] <= 500:
        clas2[2] = clas2[2] + 1
    if crla[i] > 500:
        clas2[3] = clas2[3] + 1
lab = '<1','1<X<100','100<X<500','>500'
plt.pie(clas2,labels = lab);
# Extraindo colunas de features
trainX = trainDB.iloc[:,0:(size_train[1]-2)]
# Extraindo a coluna de label
trainY = trainDB.iloc[:,(size_train[1]-2)]
# Import Select K Best and f_classif
from sklearn.feature_selection import SelectKBest,f_classif
# Import kNN classifier
from sklearn.neighbors import KNeighborsClassifier
# Utilização do algoritmo SelectKBest para determinação das 'k' colunas que fornecem melhor acurácia para o classificador
selector = SelectKBest(score_func=f_classif, k=30)
# Treinando o seletor
trainX_select = selector.fit_transform(trainX,trainY)
# Extraindo da base as 'k' colunas que foram selecionadas pelo algoritmo
ids = selector.get_support(indices = True)
trainX_tranformed = trainX.iloc[:,ids]
# Declarando classificador KNN com K = 30
knn = KNeighborsClassifier(n_neighbors=30)
# Fit do classificador
knn.fit(trainX_tranformed,trainY)
# Import bibioteca para utilização de 'Cross Validation' 
from sklearn.model_selection import cross_val_score
# Import numpy para utilizar métodos matemáticos
import numpy as np
#Realização de "Cross Validation" melhor resultado de cv
scores = cross_val_score(knn, trainX_tranformed, trainY, cv = 20)
# Media dos scores de cada bloco de dados
print(np.mean(scores))

# Este resultado é uma primeira tentativa de utilização do selectKbest
# a seguir tem-se o estudo de qual 'k' é o mais eficiente em relação
# à acurácia do classificador
# Estudo melhor 'k' para o algoritmo selectKbest
scores = []
for i in range(1,size_train[1]-1):
    selector = SelectKBest(score_func=f_classif, k=i)
    trainX_select = selector.fit_transform(trainX,trainY)
    ids = selector.get_support(indices = True)
    trainX_tranformed = trainX.iloc[:,ids]
    a = cross_val_score(knn, trainX_tranformed, trainY, cv = 20)
    scores.append(np.mean(a))
# Gráfico que relaciona a acurácia com o 'k' do algoritmo
plt.plot(np.arange(1,size_train[1]-1),scores);
plt.ylabel('Accuracy');
plt.xlabel('k-best columns');
# Tem-se que o 'k' que fornece maior acurácia é k = 8
bestK = 8
# Selecionando as colunas
selector = SelectKBest(score_func=f_classif, k=bestK)
trainX_select = selector.fit_transform(trainX,trainY)
ids = selector.get_support(indices = True)
trainX_tranformed = trainX.iloc[:,ids]
# KNN usando a base inteira
a = cross_val_score(knn, trainX, trainY, cv = 20) #CrossValidation
mean_scoresKNN = np.mean(a)
print('Acc base inteira: ',mean_scoresKNN)
print()
# KNN usando a base selecionada pelo selectKbest
a = cross_val_score(knn, trainX_tranformed, trainY, cv = 20) #CrossValidation
mean_scoresKNN = np.mean(a)
print('Acc base selecionada: ',mean_scoresKNN)
# Importando biblioteca Naive Bayes Multinomial
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
# Multinomial Naive Bayes usando a base inteira
b = cross_val_score(clf, trainX, trainY, cv = 20) #CrossValidation
mean_scoresNB = np.mean(b)
print('Acc base inteira: ',mean_scoresNB)
print()
# Multinomial Naive Bayes usando a base selecionada pelo selectKbest
b = cross_val_score(clf, trainX_tranformed, trainY, cv = 20) #CrossValidation
mean_scoresNB = np.mean(b)
print('Acc base selecionada: ',mean_scoresNB)
# Tomando as colunas selecionadas pelo algoritmo
testX = testDB.iloc[:,ids]
# fit no classificador
clf.fit(trainX_tranformed,trainY)
# Predição das labels na base de teste
predict = clf.predict(testX)
example = pd.read_csv("../input/dataset2/sample_submission_1.csv",
                     sep=r'\s*,\s*',
                     engine='python',
                     na_values="NaN")
IDs = example['Id']
submission = pd.DataFrame({'Id':IDs,'ham':predict})
submission.to_csv('submission_spam.csv',index = False)
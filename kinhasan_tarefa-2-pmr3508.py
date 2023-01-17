import pandas as pd # trabalhar os dados
import matplotlib.pyplot as plt #gerar graficos com features importantes
import numpy as np #operar vetores e matrizes
import sklearn as skl #trabalhar os dados
#lendo a base de treino
base_train= pd.read_csv("../input/pmr-3508-tarefa-2/train_data.csv", engine='python')
#Lendo a base de teste
base_test= pd.read_csv("../input/pmr-3508-tarefa-2/test_features.csv", engine='python')
#comando para imprimir a parte superior da tabela de treino para uma analise visual mais precisa
base_train.head()
base_train.info()
#determinando as dimensoes da matriz da base de dados
base_train.shape
#essa biblioteca atua junto com o matplot
import seaborn
%matplotlib inline
#visualizar a relacao entre ser ham - not spam - e os tipo de features
seaborn.pairplot(base_train,x_vars=['word_freq_credit','word_freq_free'],y_vars='ham',size=8, aspect=0.8)
#visualizar a relacao entre ser ham - not spam - e os tipo de features
seaborn.pairplot(base_train,x_vars=['word_freq_000','char_freq_$'],y_vars='ham',size=8, aspect=0.8)
from sklearn.naive_bayes import GaussianNB

features_train = base_train.drop(columns=['ham'])
target_train = base_train['ham']
gnb = GaussianNB()

gnb.fit(features_train, target_train)


from sklearn.model_selection import cross_val_score
lista1 = []
scores = cross_val_score(gnb, features_train, target_train, cv=40)
   
print(scores)
#Base de testes

reading_test = pd.read_csv("../input/pmr-3508-tarefa-2/test_features.csv")

# Realizando as predições

predictions = gnb.predict(base_test)

## A partir daqui sao comandos para entrega do trabalho
# Transformando as predictions para strings

str(predictions)

# Transformando predictions em um Panda DataFrame

df_entrega = pd.DataFrame(predictions)

# salvando as predições num arquivo CSV

df_entrega.to_csv('predictions.csv')

#amostra das predicoes
df_entrega.head()
#dimensao da matriz de predicao
df_entrega.shape
from sklearn import metrics
#Valor real
y_true = base_test

#Valor predito
y_probas = predictions

fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas, pos_label=0)

# Print Curva ROC
plt.plot(fpr,tpr)
plt.show() 

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)
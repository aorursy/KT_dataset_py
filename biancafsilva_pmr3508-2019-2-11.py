# Importando as bibliotecas necessárias



import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np
# Criando base de treino e base de testes



train_db = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=

        ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



train_db.drop(train_db.index[0],inplace=True)

#Limpando os dados faltantes:

cl_train_db = train_db.dropna()

# Visualizando a importação dos datasets:



cl_train_db.head()

# Tamanho da base:



print(cl_train_db.shape)

# Identificando as variaveis qualitativas e tratando-as:



from sklearn import preprocessing



quali_features= ['Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race',

                 'Sex', 'Country']

cl_train_db[quali_features] = cl_train_db[quali_features].apply(preprocessing.LabelEncoder().fit_transform)



# Como o banco de testes não possui o target, selecionaremos uma parte do banco de dados de treino

# para testar os algoritmos



from sklearn.model_selection import train_test_split



train_x = cl_train_db.iloc[:,0:-1]

train_y = cl_train_db.Target



# Usaremos 80% do banco para treino e 20% para teste

train_x, test_x, train_y, test_y = train_test_split(train_x,train_y,test_size=0.20)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

model_log = LogisticRegression()

model_log.fit(train_x,train_y)
print("Métricas da regressão logística:")

print(confusion_matrix(test_y, model_log.predict(test_x)))

print(classification_report(test_y, model_log.predict(test_x)))

print('Accuracy: ',accuracy_score(test_y, model_log.predict(test_x)))
from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



model_arv = DecisionTreeClassifier()

model_arv.fit(train_x,train_y)

y_pred=model_arv.predict(test_x)

scores = cross_val_score(model_arv, train_x, train_y, cv=5)

media_arv=np.mean(scores)



y_pred_arv=model_arv.predict(train_x)

ac_arv=accuracy_score(train_y,y_pred_arv,normalize=True,sample_weight=None)



print("Média dos valores obtidos com validação cruzada:")

print(media_arv)

print("Acurácia do modelo:")

print(ac_arv)
from sklearn.svm import SVC

svm = SVC()

svm.fit(train_x,train_y)

svm_pred = svm.predict(test_x)

                      

print("Métricas do SVM")

print(confusion_matrix(test_y, svm_pred))

print(classification_report(test_y, svm_pred))

print('Accuracy: ',accuracy_score(test_y, svm_pred))
# Importa as bibliotecas necessárias para realizar os treinos e testes dos arquivos. .

import numpy as np 

import pandas as pd

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) # Chamada para manipular informações externas
# Lê o arquivo de treino e exibe em tela:

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
# Lê o arquivo de teste e exibe em tela:

teste = pd.read_csv("../input/test.csv")

print(teste.shape)

teste.head()
train.columns # exibe as colunas do arquivo de treino
train_copy = train # Faz uma cópia do arquivo de treino para que não seja modificado os dados iniciais.

train_copy = train_copy.replace(-1, np.NaN) #Verifica as colunas que estão nulas e altera para -1

test_copy = teste # Faz uma cópia do arquivo de teste para que não seja modificado os dados iniciais.

test_copy = test_copy.replace(-1, np.NaN) #Verifica as colunas que estão nulas e altera para -1
import missingno as msno # Plota gráfico de treino pelas features.

%matplotlib inline

msno.bar(train_copy)
msno.bar(test_copy) # Plota o gráfico de teste pelas features.
# Separando parte dos dados para amostra de treinamento.

from sklearn.model_selection import train_test_split



X_train = train.drop(['target'], axis=1).values

y_train = train['target'].values

X_train_main, X_train_validate, y_train_main, y_train_validate = train_test_split(X_train,y_train,test_size=0.5,stratify=y_train) 
# Encontrando recurso de importação usando o ExtraTreeClassifier 

from sklearn.ensemble import ExtraTreesClassifier 

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



forest = ExtraTreesClassifier(n_estimators=250,

                              random_state=0)

forest.fit(X_train_main, y_train_main)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_train_main.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure(figsize=(20,10))

plt.title("Feature importances")

plt.bar(range(X_train_main.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(X_train_main.shape[1]), indices)

plt.xlim([-1, X_train_main.shape[1]])

plt.show()
# Buscamos apenas os 28 primeiros registros, conforme o gráfico.

important_feature = []

for f in range(28):

    important_feature.append(indices[f])

print(important_feature)
# Criando um dataframe apenas com os dados que foram considerados como importantes.

train_copy = train.drop(['target'],axis=1)

final_train = train_copy.iloc[:,important_feature]

X_train = final_train.values

y_train = train['target'].values

# final_train = train.iloc[:,important_feature]

# print(final_train.head())

# X_train = final_train.drop(['target'], axis=1).values

# y_train = final_train['target'].values

X_train_main, X_train_validate, y_train_main, y_train_validate = train_test_split(X_train,y_train,test_size=0.2,stratify=y_train) 
# Aplicando o modelo RandomForestClassifier 

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train_main, y_train_main)
#Treinando o modelo.

predicted_train_validate = clf.predict(X_train_validate)

actual_train_validate = y_train_validate
#Verificando a acuracidade. 

from sklearn.metrics import accuracy_score

accuracy_score(actual_train_validate, predicted_train_validate)
# Verificando e preparando o arquivo para a saída do arquivo.

test_copy = teste.iloc[:,important_feature]

X_test = test_copy.values

predicted_test = clf.predict(X_test)
Final = pd.DataFrame({'id': teste['id'].values, 'target': predicted_test})
print(Final)
Final.to_csv("submission_output.csv", index=False) 
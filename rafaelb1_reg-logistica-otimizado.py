import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use("seaborn-muted")

%matplotlib inline
treino = pd.read_csv("../input/titanic/train.csv")

teste = pd.read_csv("../input/titanic/test.csv")
treino.head()
treino.info()
survived = treino["Survived"]



survived
treino.drop(["Survived"], axis = 1, inplace = True)



treino_index = treino.shape[0]

teste_index = teste.shape[0]



banco_geral = pd.concat(objs = [treino, teste], axis = 0).reset_index(drop = True)
banco_geral.head()
banco_geral.shape
banco_geral = banco_geral.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1)
banco_geral.head()
banco_geral["Age"] = pd.cut(banco_geral["Age"], bins = [0, 20, 40, 60, 80], labels = ["0-20", "21-40", "41-60", "61-80"], include_lowest = True)



banco_geral["Age"] = banco_geral["Age"].values.add_categories("SI").fillna("SI")
banco_geral["Age"].value_counts()
banco_geral["Pclass"] = banco_geral.replace({'Pclass': {1: "Classe 1", 

                                                        2: "Classe 2", 

                                                        3: "Classe 3"}})
banco_geral["Pclass"].value_counts()
moda_fare = banco_geral["Fare"].mode()[0]



print("A moda das Tarifas é", moda_fare)
banco_geral["Fare"] = banco_geral["Fare"].fillna(moda_fare)



banco_geral["Fare"]
banco_geral = banco_geral.replace({"Fare": {0 : 1}})
banco_geral["Fare"] = np.log(banco_geral["Fare"])



banco_geral.head()
banco_geral["Embarked"] = banco_geral["Embarked"].fillna("SI")
banco_geral["Embarked"].value_counts()
banco_geral["Qtd_familiares"] = banco_geral["SibSp"] + banco_geral["Parch"]



banco_geral["Qtd_familiares"]



banco_geral = banco_geral.drop(["SibSp"], axis = 1)
banco_geral = pd.get_dummies(banco_geral)
banco_geral.head()
treino1 = banco_geral.iloc[:treino_index]



print("Dimensões do novos dados de treino")



treino1.shape
teste1 = banco_geral.iloc[:teste_index]



print("Dimensões do novos dados de teste")



teste1.shape
treino1 = treino1.assign(Survived = survived)



treino1.head()
from sklearn.model_selection import train_test_split



x_treino, x_valid, y_treino, y_valid = train_test_split(treino1.drop("Survived", axis = 1), treino1["Survived"], train_size = 0.5, random_state = 1234)
print("Os dados de treino possui dimensões:", treino1.shape)

print("---")

print("x_treino possui dimensões:", x_treino.shape)

print("---")

print("y_treino possui dimensões:", y_treino.shape)

print("---")

print("x_valid possui dimensões:", x_valid.shape)

print("---")

print("y_valid possui dimensões:", y_valid.shape)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



logreg = LogisticRegression(random_state = 1234, class_weight = {0 : 0.6, 1 : 0.4})

logreg.fit(x_treino, y_treino)
previsao = logreg.predict(x_valid)



previsao[:6]
from sklearn.metrics import confusion_matrix



confusion_matrix = confusion_matrix(y_valid, previsao)



print(confusion_matrix)
melhora_piora_modelo = logreg.score(x_valid, y_valid) - 0.8071748878923767



if melhora_piora_modelo > 0:

    

    print("Temos um ganho de", melhora_piora_modelo, "na acurácia.")



elif melhora_piora_modelo < 0:

    

    print("Temos uma perda de", melhora_piora_modelo, "na acurácia.")

    

else:

    

    print("Não temos ganho")
x_treino_retrain = treino1.drop("Survived", axis = 1)

y_treino_retrain = treino1["Survived"]



logreg_retrain = LogisticRegression(random_state = 1234)



logreg_retrain.fit(x_treino_retrain, y_treino_retrain)
previsoes_subm = logreg_retrain.predict(teste1)



previsoes_subm.shape
id_teste = teste["PassengerId"]



subm = pd.DataFrame()



subm["PassengerId"] = id_teste



subm["Survived"] = previsoes_subm
subm.head()
# subm.to_csv("subm6.csv", index = False)
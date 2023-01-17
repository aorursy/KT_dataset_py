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
plt.figure(figsize = [10, 6])



treino["Survived"].value_counts(normalize = True).mul(100).plot(kind = "bar", color = "orangered", edgecolor = "black")

plt.title("Percentual de passageiros que sobreviveram ou não ao Titanic", fontsize = 14, color = "black")

plt.xlabel("Sobreviveu", fontsize = 14, color = "black")

plt.ylabel("Percentual", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)

plt.show()
pd.concat([treino["Survived"].value_counts(), 

                treino["Survived"].value_counts(normalize = True).mul(100).round(2)],axis = 1, keys = ("Quantidade", "Percentual"))
plt.figure(figsize = [15, 9])



#---



plt.subplot(3, 2, 1)



treino["Pclass"].value_counts(normalize = True, ascending = True).mul(100).plot(kind = "barh", color = "orangered", edgecolor = "black")

plt.xlabel("Percentual", fontsize = 14, color = "black")

plt.ylabel("Classe do passageiro", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)



plt.subplot(3, 2, 2)



treino["Sex"].value_counts(normalize = True, ascending = True).mul(100).plot(kind = "barh", color = "orangered", edgecolor = "black")

plt.xlabel("Percentual", fontsize = 14, color = "black")

plt.ylabel("Sexo", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)



plt.subplot(3, 2, 3)



treino["Embarked"].value_counts(normalize = True, ascending = True).mul(100).plot(kind = "barh", color = "orangered", edgecolor = "black")

plt.xlabel("Percentual", fontsize = 14, color = "black")

plt.ylabel("Porto embarcado", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)



plt.subplot(3, 2, 4)



treino["SibSp"].value_counts(normalize = True, ascending = True).mul(100).plot(kind = "barh", color = "orangered", edgecolor = "black")

plt.xlabel("Percentual", fontsize = 14, color = "black")

plt.ylabel("Nº de irmãos/conjuge\n a bordo", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)



plt.subplot(3, 2, 5)



treino["Parch"].value_counts(normalize = True, ascending = True).mul(100).plot(kind = "barh", color = "orangered", edgecolor = "black")

plt.xlabel("Percentual", fontsize = 14, color = "black")

plt.ylabel("Nº de pais/filhos\n a bordo", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)



        



#---



plt.subplots_adjust(hspace = 0.5)

plt.show()
plt.figure(figsize = [10, 6])



#---



plt.subplot(2, 1, 1)



sns.distplot(treino["Age"], color = "orangered")

plt.xlabel("Idade", fontsize = 14, color = "black")

plt.ylabel("Densidade", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")



plt.subplot(2, 1, 2)



sns.distplot(treino["Fare"], color = "orangered")

plt.xlabel("Tarifa", fontsize = 14, color = "black")

plt.ylabel("Densidade", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")





plt.subplots_adjust(hspace = 0.5)

plt.show()
treino[["Name", "Ticket", "Cabin"]].head()
#--- Sobreviveu x Classe



plt.figure(figsize = [10, 6])



treino[["Survived", "Pclass"]].reset_index().groupby(["Survived", "Pclass"], as_index = False).size().unstack().plot.bar()

plt.xlabel("Sobreviveu", fontsize = 14, color = "black")

plt.ylabel("Quantidade", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)

plt.show()
#--- Sobreviveu x Sexo



# Hipótese: Qual gênero tem mais chances de sobreviver ao desastre ?



treino[["Survived", "Sex"]].reset_index().groupby(["Survived", "Sex"], as_index = False).size().unstack().plot.bar()

plt.xlabel("Sobreviveu", fontsize = 14, color = "black")

plt.ylabel("Quantidade", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)

plt.show()
#--- Sobreviveu x Porto embarcado



treino[["Survived", "Embarked"]].reset_index().groupby(["Survived", "Embarked"], as_index = False).size().unstack().plot.bar()

plt.xlabel("Sobreviveu", fontsize = 14, color = "black")

plt.ylabel("Quantidade", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)

plt.show()
#--- Sobreviveu x Nº de irmãos/conjuge a bordo



treino[["SibSp", "Survived"]].reset_index().groupby(["SibSp", "Survived"], as_index = False).size().unstack().plot.bar()

plt.xlabel("Nº de irmãos/conjuge a bordo", fontsize = 14, color = "black")

plt.ylabel("Quantidade", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)

plt.show()
#--- Sobreviveu x Nº de pais/filhos a bordo



treino[["Parch", "Survived"]].reset_index().groupby(["Parch", "Survived"], as_index = False).size().unstack().plot.bar()

plt.xlabel("Nº de pais/filhos a bordo", fontsize = 14, color = "black")

plt.ylabel("Sobreviveu", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)

plt.show()
sns.boxplot(x = treino["Survived"], y = treino["Age"], data = treino, palette = "Dark2")

plt.xlabel("Sobreviveu", fontsize = 14, color = "black")

plt.ylabel("Idade", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)



plt.show()
sns.boxplot(x = treino["Survived"], y = treino["Fare"], data = treino, palette = "Dark2")

plt.xlabel("Sobreviveu", fontsize = 14, color = "black")

plt.ylabel("Tarifa paga", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)



plt.show()
sns.boxplot(x = "Survived", y = "Fare", hue = "Pclass",

                 data = treino, palette = "Dark2")

plt.xlabel("Sobreviveu", fontsize = 14, color = "black")

plt.ylabel("Tarifa", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.xticks(rotation = 0)



plt.show()
survived = treino["Survived"]



survived
treino.drop(["Survived"], axis = 1, inplace = True)



treino_index = treino.shape[0]

teste_index = teste.shape[0]



banco_geral = pd.concat(objs = [treino, teste], axis = 0).reset_index(drop = True)
banco_geral.head()
banco_geral.shape
miss_val_treino = banco_geral.isnull().sum()



miss_val_treino = miss_val_treino[miss_val_treino > 0]



dados_miss_val_treino = pd.DataFrame(miss_val_treino)



dados_miss_val_treino = dados_miss_val_treino.reset_index().sort_values(by = 0, ascending = True)



dados_miss_val_treino

dados_miss_val_treino.columns = ["Variável", "Quantidade"]







plt.figure(figsize = [10, 6])

plt.barh(dados_miss_val_treino["Variável"], dados_miss_val_treino["Quantidade"], align = "center", color = "orangered")

plt.xlabel("Quantidade de valores faltantes", fontsize = 14, color = "black")

plt.ylabel("Variável", fontsize = 14, color = "black")

plt.tick_params(axis = "x", labelsize = 12, labelcolor = "black")

plt.tick_params(axis = "y", labelsize = 12, labelcolor = "black")

plt.show()
banco_geral = banco_geral.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1)
banco_geral.head()
banco_geral["Pclass"] = banco_geral["Pclass"].astype(str)
banco_geral["Age"] = banco_geral["Age"].fillna(banco_geral["Age"].mode()[0])
banco_geral["Embarked"] = banco_geral["Embarked"].fillna("SI")
banco_geral["Fare"] = banco_geral["Fare"].fillna(banco_geral["Fare"].mode()[0])



banco_geral.isnull().sum()
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

print("----" * 15)

print("x_treino possui dimensões:", x_treino.shape)

print("----" * 15)

print("y_treino possui dimensões:", y_treino.shape)

print("----" * 15)

print("x_valid possui dimensões:", x_valid.shape)

print("----" * 15)

print("y_valid possui dimensões:", y_valid.shape)
x_treino1 = x_treino[["Age", "SibSp", "Parch", "Fare", "Pclass_2", "Pclass_3", "Sex_female", "Embarked_C", "Embarked_Q", "Embarked_SI"]]



x_treino1["intercepto"] = 1.0
import statsmodels.api as sm



log_mod_sm = sm.Logit(y_treino, x_treino1).fit()



print(log_mod_sm.summary())
odds_ratio_banco = pd.DataFrame(np.exp(log_mod_sm.params).round(2)).reset_index()

odds_ratio_banco.columns = ["Variável", "Odds ratio"]



var_retirar = ["Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_SI"]



odds_ratio_banco.loc[~ odds_ratio_banco['Variável'].isin(var_retirar)]
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



reg_log = LogisticRegression(max_iter = 500, random_state = 1234)

reg_log.fit(x_treino, y_treino)
previsao = reg_log.predict(x_valid)



previsao[:6]
from sklearn.metrics import confusion_matrix



confusion_matrix = confusion_matrix(y_valid, previsao)



print(confusion_matrix)
reg_log.score(x_valid, y_valid)
from sklearn.metrics import classification_report



print(classification_report(y_valid, previsao))
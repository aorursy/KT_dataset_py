import pandas as pd # para Processamento de Dados
import matplotlib.pyplot as plt #Para plotagem de gráficos
import numpy as np # Algebra Linear
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
adult = pd.read_csv("../input/adultdata/train_data.csv", na_values="?")
adult
adult["workclass"].value_counts()
adult = pd.get_dummies(adult, columns=["workclass"])
# "Quebrando" a coluna "workclass" em subtópicos com valores numéricos para o cálculo de distância euclidiana
adult
adult["workclass_others"] = adult["workclass_Without-pay"] + adult["workclass_Never-worked"] + adult["workclass_Self-emp-inc"]
#Agrupa as "workclass" menos frequentes na base de dados em "workclass_others"
adult = adult.drop(["workclass_Never-worked", "workclass_Self-emp-inc", "workclass_Without-pay"], axis = 1)
#Remove as "workclass" menos frequentes (que foram agrupadas acima em "others")
adult["workclass_gov"] = adult["workclass_State-gov"] + adult["workclass_Local-gov"] + adult["workclass_Federal-gov"]
# Agrupa "workclass" relacionadas ao setor público
adult = adult.drop(["workclass_State-gov", "workclass_Local-gov", "workclass_Federal-gov"], axis=1)
# Remove as "workclass" que foram agrupadas
adult = adult.drop(["native.country", "sex", "Id", "fnlwgt", "capital.gain", "education", "relationship", "marital.status"], axis=1)
# Remove alguns features pouco relacionados com a receita que podem gerar overfitting
adult["race"].value_counts()
adult["occupation"].value_counts()
adult = pd.get_dummies(adult, columns=["race", "occupation"])
# Fragmentando os features "race" e "occupation" em features numéricos como feito acima para o feature "workclass"
adult["race_Other"] = adult["race_Other"] + adult["race_Amer-Indian-Eskimo"]
adult["occupation_Other-service"] += adult["occupation_Armed-Forces"] + adult["occupation_Priv-house-serv"] + adult["occupation_Protective-serv"]
adult = adult.drop(["occupation_Armed-Forces", "occupation_Priv-house-serv", "occupation_Protective-serv"], axis=1)
adultX = adult.drop("income", axis=1)
adultY = adult.income
adultTestX = pd.read_csv("../input/adultdata/test_data.csv", na_values="?")
#adult.drop(["income"], axis=1)
adultTestX = pd.get_dummies(adultTestX, columns=["workclass"])
adultTestX["workclass_others"] = adultTestX["workclass_Without-pay"] + adultTestX["workclass_Never-worked"] + adultTestX["workclass_Self-emp-inc"]
adultTestX = adultTestX.drop(["workclass_Never-worked", "workclass_Self-emp-inc", "workclass_Without-pay"], axis = 1)
adultTestX["workclass_gov"] = adultTestX["workclass_State-gov"] + adultTestX["workclass_Local-gov"] + adultTestX["workclass_Federal-gov"]
adultTestX = adultTestX.drop(["workclass_State-gov", "workclass_Local-gov", "workclass_Federal-gov"], axis=1)
adultTestX = adultTestX.drop(["native.country", "sex", "Id", "fnlwgt", "capital.gain", "education", "relationship", "marital.status"], axis=1)
adultTestX = pd.get_dummies(adultTestX, columns=["race", "occupation"])
adultTestX["race_Other"] = adultTestX["race_Other"] + adultTestX["race_Amer-Indian-Eskimo"]
adultTestX["occupation_Other-service"] += adultTestX["occupation_Armed-Forces"] + adultTestX["occupation_Priv-house-serv"] + adultTestX["occupation_Protective-serv"]
adultTestX = adultTestX.drop(["occupation_Armed-Forces", "occupation_Priv-house-serv", "occupation_Protective-serv"], axis=1)
kNN = KNeighborsClassifier(n_neighbors=4)
scores = cross_val_score(kNN, adultX, adultY, cv=10)
scores
scores.mean()
kNN.fit(adultX, adultY)
predictions = kNN.predict(adultTestX)
df = pd.DataFrame(predictions)
df.to_csv('predictions.csv')
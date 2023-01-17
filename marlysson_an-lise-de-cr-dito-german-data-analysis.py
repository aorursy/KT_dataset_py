import pandas as pd
atributos = ["montante", "duracao", "historico_credito", 
              "proposito", "montante_credito", "poupanca",
              "tempo_empregado","taxa_parcelamento",
              "estado_civil_sexo","tipo_participacao_credito", 
              "tempo_moradia", "propriedade","idade",
              "gastos_adicionais", "habitacao","quantidade_creditos","emprego",
              "dependentes","telefone","trabalhador_estrangeiro","risco"]
df = pd.read_csv("../input/credit_approval.txt",header=None, sep=" ",names=atributos)
df.head(3)
codigos_historico_de_creditos = {
    "A30": "no credits taken/all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly till now",
    "A33": "delay in paying off in the past",
    "A34": "critical account/other credits existing (not at this bank)"
}

codigos_proposito = {
    "A40": "car(new)",
    "A41": "car(used)",
    "A42": "furniture/equipment",
    "A43": "radio/television",
    "A44": "domestic appliances",
    "A45": "repairs",
    "A46": "education",
    "A47": "vacation",
    "A48": "retraining",
    "A49": "business",
    "A410": "others"
}

codigo_estado_civil_sexo = {
    "A91": "male : divorced/separated",
    "A92": "female : divorced/separated/married",
    "A93": "male : single",
    "A94": "male : married/windowed",
    "A95": "female : single"
}

codigos_outros_devedores = {
    "A101": None,
    "A102": "co-applicant",
    "A103": "guarantor",
}

codigos_propriedade = {
    "A121": "real state",
    "A122": "building society/life insurance",
    "A123": "car",
    "A124": "unknown/no property"
}

codigos_planos_de_parcelamento = {
    "A141": "bank",
    "A142": "stores",
    "A143": "None"
}

codigos_residencia = {
    "A151": "rent",
    "A152": "own",
    "A153": "for free"
}

codigos_estado_emprego = {
    "A171": "unemployed/unskilled-non-resident",
    "A172": "unskilled-resident",
    "A173": "skilled employee/official",
    "A174": "management/self-employed/highly qualified employee/officer"
}

codigos_telefone = {
    "A191": None,
    "A192": "yes"
}

codigos_trabalhador_estrangeiro = {
    "A201": "yes",
    "A202": "no"
}
codigos_status_atual_conta_corrente = {
    "A11": "< 0",
    "A12": "< 199",
    "A13": ">= 200",
    "A14": None
}

codigos_reserva_poupanca = {
    "A61": "< 100",
    "A62": "< 499",
    "A63": "< 999",
    "A64": ">= 1000",
    "A65": "unknown"
}

codigos_tempo_emprego = {
    "A71": None,
    "A72": "< 1", # Menos de 1 ano
    "A73": "< 4", # Entre 1 ano e menos que 4 anos
    "A74": "< 7", # Entre 4 anos e menos que 7 anos
    "A75": ">= 7" # Mais de 7 anos
}
colunas_para_codigos = {
    "montante"             : codigos_status_atual_conta_corrente,
    "historico_credito"    : codigos_historico_de_creditos,
    "proposito"            : codigos_proposito,
    "poupanca"             : codigos_reserva_poupanca,
    "tempo_empregado"      : codigos_tempo_emprego,
    "estado_civil_sexo"    : codigo_estado_civil_sexo, 
    "tipo_participacao_credito"     : codigos_outros_devedores,
    "propriedade"          : codigos_propriedade,
    "gastos_adicionais": codigos_planos_de_parcelamento,
    "habitacao"            : codigos_residencia,
    "emprego"              : codigos_estado_emprego,
    "telefone"             : codigos_telefone,
    "trabalhador_estrangeiro"  : codigos_trabalhador_estrangeiro
}
df.replace(colunas_para_codigos,inplace=True)
df.head(3)
df.replace({"unknown":None},inplace=True)
df.dtypes
df.info()
def criar_sexo_e_estado_civil(coluna):
    dados_separados = coluna.split(":")
    
    sexo = dados_separados[0].strip()
    estado_civil = dados_separados[1].strip()

    return pd.Series([sexo,estado_civil])
df[["sexo","estado_civil"]] = df["estado_civil_sexo"].apply(criar_sexo_e_estado_civil)
colunas = ["telefone","trabalhador_estrangeiro", "estado_civil_sexo","gastos_adicionais","tipo_participacao_credito"]
df = df.drop(colunas,axis=1)
df.dtypes

import seaborn as sns
import matplotlib.pyplot as plt
a = sns.countplot(x="montante",data=df)
a.set_title("Contagem do montante por tipo")
plt.figure(figsize=(15, 5))
sns.heatmap(df.corr(),annot=True)
plt.figure(figsize=(15, 5))
sns.swarmplot(x="historico_credito",y="montante_credito",data=df)
plt.figure(figsize=(15, 5))
sns.distplot(df.idade)
df.head()
def mapear_valores(coluna):
    valores = tuple(set(df[coluna].values))

    associados = tuple(range(len(valores)))

    df[coluna].replace(valores,associados,inplace=True)
colunas = ["historico_credito","montante","proposito","poupanca",
           "tempo_empregado","propriedade","habitacao",
           "emprego","sexo","estado_civil"]

for coluna in colunas:
    mapear_valores(coluna)
from sklearn.model_selection import train_test_split
x = df.drop('risco', 1).values
y = df["risco"].values

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)
def aplicar_modelo(modelo, x_treino, y_treino, x_teste, y_teste):
    
    modelo.fit(x_treino,y_treino)
    
    risco = modelo.predict(x_teste)
    
    return accuracy_score(y_teste,risco)
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

naive = MultinomialNB()

resultado = aplicar_modelo(naive,x_treino,y_treino, x_teste,y_teste)
print("Naive Bayes: {}".format(resultado))
from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier()

resultado = aplicar_modelo(ada_boost,x_treino,y_treino, x_teste,y_teste)
print("Ada boost: {}".format(resultado))
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()

resultado = aplicar_modelo(random_forest,x_treino,y_treino, x_teste,y_teste)
print("Random Forest: {}".format(resultado))
from sklearn.linear_model import LogisticRegression

logistic_regression = LogisticRegression()

resultado = aplicar_modelo(logistic_regression,x_treino,y_treino, x_teste,y_teste)
print("Regressão Logística: {}".format(resultado))
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
one_vs_one = OneVsOneClassifier(LinearSVC(random_state = 0))

resultado = aplicar_modelo(one_vs_one,x_treino,y_treino, x_teste,y_teste)
print("One vs One classifier: {}".format(resultado))
from sklearn.multiclass import OneVsRestClassifier

one_vs_rest = OneVsRestClassifier(LinearSVC(random_state = 0))

resultado = aplicar_modelo(one_vs_rest,x_treino,y_treino, x_teste,y_teste)
print("One vs Rest classifier : {}".format(resultado))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

resultado = aplicar_modelo(knn,x_treino,y_treino, x_teste,y_teste)
print("KNN classifier: {}".format(resultado))
from sklearn.cross_validation import cross_val_score
import numpy as np

algoritmos = [MultinomialNB(), AdaBoostClassifier(), 
              RandomForestClassifier(), LogisticRegression(), 
              OneVsOneClassifier(LinearSVC(random_state = 0)),
              OneVsRestClassifier(LinearSVC(random_state = 0)),
              KNeighborsClassifier()
             ]

resultados = []

k_folding = len(df.columns) // 2

for modelo in algoritmos:
    
    resultado = cross_val_score(modelo,x_treino,y_treino,cv=k_folding)
    resultados.append(np.mean(resultado))
    
    print("Algoritmo: {}\n Resultado: {:.2f}\n".format(str(modelo.__class__).split(".")[-1], np.mean(resultado)))
resultados_series = pd.Series(resultados, index=['Naive Bayes','AdaBoostClassifier',
                                       'RandomForestClassifier','LogisticRegression',
                                       'OneVsOneClassifier','OneVsRestClassifier',
                                      'KNeighborsClassifier'])
resultados_series.plot(kind="bar")

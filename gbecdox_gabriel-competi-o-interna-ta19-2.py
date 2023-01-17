import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
treino = pd.read_csv("../input/ta192/train.csv")    # Dataset que será utilizado, futuramente, para treinar o modelo.

treino.set_index("ID", inplace=True) # Define a coluna ID como índice (evita que essa interfira na predição).
pd.set_option('display.max_columns', 25) # Permitir a visualização de todas as 25 colunas.
treino.rename({"PAY_0":"PAY_1"}, axis="columns", inplace=True) # Altera o nome da coluna PAY_0 para PAY_1
treino.info()
print("SEX:", treino.SEX.unique())

print("EDUCATION:", treino.EDUCATION.unique())

print("MARRIAGE:", treino.MARRIAGE.unique())
# Substitui "male" e "m" por 1.

treino["SEX"].replace(to_replace="male", value=1, inplace=True)

treino["SEX"].replace(to_replace="m", value=1, inplace=True)



# Substitui "female" e "f" por 2.

treino["SEX"].replace(to_replace="female", value=2, inplace=True)

treino["SEX"].replace(to_replace="f", value=2, inplace=True)
# Substitui "graduate school", "university", "high school" e "others" respectivamente por 1, 2, 3 e 4.

treino["EDUCATION"].replace(to_replace="graduate school", value=1, inplace=True)

treino["EDUCATION"].replace(to_replace="university", value=2, inplace=True)

treino["EDUCATION"].replace(to_replace="high school", value=3, inplace=True)

treino["EDUCATION"].replace(to_replace="others", value=4, inplace=True)
# Substitui "married" e "MARRIED" por 1.

treino["MARRIAGE"].replace(to_replace="married", value=1, inplace=True)

treino["MARRIAGE"].replace(to_replace="MARRIED", value=1, inplace=True)



# Substitui "single" e "SINGLE" por 2.

treino["MARRIAGE"].replace(to_replace="single", value=2, inplace=True)

treino["MARRIAGE"].replace(to_replace="SINGLE", value=2, inplace=True)



# Substitui "others" e "OTHERS" por 3.

treino["MARRIAGE"].replace(to_replace="others", value=3, inplace=True)

treino["MARRIAGE"].replace(to_replace="OTHERS", value=3, inplace=True)
treino.describe()
print("Número de valores faltantes:", treino.shape[0]*treino.shape[1] - treino.count().sum(), end=" ")

print("(", "{0:.2f}".format((treino.shape[0]*treino.shape[1] - treino.count().sum())*100/(treino.shape[0]*treino.shape[1])), "%)", sep="")
print("Número de linhas com valores faltantes:", (treino.shape[0] - treino.dropna().shape[0]), end=" ")

print("(", "{0:.2f}".format((treino.shape[0] - treino.dropna().shape[0])*100/treino.shape[0]), "%)", sep="")
treino["AGE"].value_counts()
treino["AGE"].values[treino['AGE'] >= 90] = 77 # Trasnforma idades maiores ou iguais a 90 anos para 77 anos.

treino["AGE"].values[treino['AGE'] < 20] = 20  # Trasnforma idades menores do que 20 anos para 20 anos.
sns.distplot(treino["AGE"].dropna())
# Importa a biblioteca utilizada para criar os valores aleatórios.

import random



# Garante a reprodutibilidade dos resultados.

random.seed(101)



# Substitui os valores faltantes.

treino["AGE"].fillna(treino["AGE"].apply(lambda v: random.randint(20,80)), inplace=True)
sns.distplot(treino["AGE"])
treino["SEX"].fillna(1.5, inplace=True)
treino["MARRIAGE"].fillna(3, inplace=True)
treino["EDUCATION"].fillna(4, inplace=True)
treino[["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_1"]].head(20)
# Substitui os valores faltantes da coluna PAY_1 pela moda dessa.

treino["PAY_1"].fillna(int(treino["PAY_1"].mode()), inplace=True)



# Itera por cada linha das colunas de PAY_2 até PAY_6 aplicando a transformação descrita.

for x in range(2, 7):    

    treino["PAY_" + str(x)] = treino.apply(lambda linha: linha["PAY_" + str(x-1)] if np.isnan(linha["PAY_" + str(x)]) else linha["PAY_" + str(x)], axis=1)
treino[["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]].head(30)
# Itera por cada linha das colunas de BILL_AMT1 até BILL_AMT6 aplicando a transformação descrita.

for x in range(1, 7):

    treino["BILL_AMT" + str(x)] = treino.apply(

        lambda linha: int(linha[["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]].mean(axis=0)) 

        if np.isnan(linha["BILL_AMT" + str(x)]) 

        else linha["BILL_AMT" + str(x)], axis=1)
treino[["PAY_AMT6", "PAY_AMT5", "PAY_AMT4", "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]].head(30)
# Itera por cada linha das colunas de PAY_AMT1 até PAY_AMT6 aplicando a transformação descrita.

for x in range(1, 7):

    treino["PAY_AMT" + str(x)] = treino.apply(

        lambda linha: int(linha[["PAY_AMT6", "PAY_AMT5", "PAY_AMT4", "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]].mean(axis=0)) 

        if np.isnan(linha["PAY_AMT" + str(x)]) 

        else linha["PAY_AMT" + str(x)], axis=1)
print("LIMIT_BAL:", sorted(treino.LIMIT_BAL.unique()))
plt.figure(figsize=(8, 6))



sns.distplot(treino["LIMIT_BAL"].dropna())
plt.figure(figsize=(8, 6))



sns.boxplot(treino["LIMIT_BAL"].dropna())
# Garante a reprodutibilidade dos resultados.

random.seed(101)



# Substitui os valores faltantes.

treino["LIMIT_BAL"].fillna(treino["LIMIT_BAL"].apply(lambda x: random.randrange(10000, 500001, 10000)), inplace=True)
plt.figure(figsize=(8, 6))



sns.distplot(treino["LIMIT_BAL"])
plt.figure(figsize=(8,6))



sns.boxplot(treino["LIMIT_BAL"])
treino.info()
treino.describe()
teste_ID = pd.read_csv("../input/ta192/test.csv")  # Dataset de teste com ID, será usado somente no final para extrair o ID.

teste = teste_ID.drop("ID", axis=1) # Dataset de teste sem ID, será tratado e utilizado "de fato" (para predições e etc.).
teste.rename({"PAY_0":"PAY_1"}, axis="columns", inplace=True) # Altera o nome da coluna PAY_0 para PAY_1.
teste.info()
teste.describe()
# Substitui "male" e "m" por 1.

teste["SEX"].replace(to_replace="male", value=1, inplace=True)

teste["SEX"].replace(to_replace="m", value=1, inplace=True)



# Substitui "female" e "f" por 2.

teste["SEX"].replace(to_replace="female", value=2, inplace=True)

teste["SEX"].replace(to_replace="f", value=2, inplace=True)
# Substitui "graduate school", "university", "high school" e "others" respectivamente por 1, 2, 3 e 4.

teste["EDUCATION"].replace(to_replace="graduate school", value=1, inplace=True)

teste["EDUCATION"].replace(to_replace="university", value=2, inplace=True)

teste["EDUCATION"].replace(to_replace="high school", value=3, inplace=True)

teste["EDUCATION"].replace(to_replace="others", value=4, inplace=True)
# Substitui "married" e "MARRIED" por 1.

teste["MARRIAGE"].replace(to_replace="married", value=1, inplace=True)

teste["MARRIAGE"].replace(to_replace="MARRIED", value=1, inplace=True)



# Substitui "single" e "SINGLE" por 2.

teste["MARRIAGE"].replace(to_replace="single", value=2, inplace=True)

teste["MARRIAGE"].replace(to_replace="SINGLE", value=2, inplace=True)



# Substitui "others" e "OTHERS" por 3.

teste["MARRIAGE"].replace(to_replace="others", value=3, inplace=True)

teste["MARRIAGE"].replace(to_replace="OTHERS", value=3, inplace=True)
teste["AGE"].values[teste_ID['AGE'] >= 90] = 77

teste["AGE"].values[teste_ID['AGE'] < 20] = 20
# Importa a biblioteca utilizada para criar os valores aleatórios.

import random



# Garante a reprodutibilidade dos resultados.

random.seed(101)



# Substitui os valores faltantes de AGE.

teste["AGE"].fillna(teste["AGE"].apply(lambda v: random.randint(20, 80)), inplace=True)
# Preenche valores faltantes de SEX, MARRIAGE e EDUCATION.

teste["SEX"].fillna(1.5, inplace=True)

teste["MARRIAGE"].fillna(3, inplace=True)

teste["EDUCATION"].fillna(4, inplace=True)
# Substitui os valores faltantes da coluna PAY_1 pela moda dessa.

teste["PAY_1"].fillna(int(teste["PAY_1"].mode()), inplace=True)



# Itera por cada linha das colunas de PAY_2 até PAY_6 aplicando a transformação descrita anteriormente.

for x in range(2, 7):    

    teste["PAY_" + str(x)] = teste.apply(lambda linha: linha["PAY_" + str(x-1)] if np.isnan(linha["PAY_" + str(x)]) else linha["PAY_" + str(x)], axis=1)
# Itera por cada linha das colunas de BILL_AMT1 até BILL_AMT6 aplicando a transformação descrita anteriormente.

for x in range(1, 7):

    teste["BILL_AMT" + str(x)] = teste.apply(

        lambda linha: int(linha[["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]].mean(axis=0)) 

        if np.isnan(linha["BILL_AMT" + str(x)]) 

        else linha["BILL_AMT" + str(x)], axis=1)
# Itera por cada linha das colunas de PAY_AMT1 até PAY_AMT6 aplicando a transformação descrita anteriormente.

for x in range(1, 7):

    teste["PAY_AMT" + str(x)] = teste.apply(

        lambda linha: int(linha[["PAY_AMT6", "PAY_AMT5", "PAY_AMT4", "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]].mean(axis=0)) 

        if np.isnan(linha["PAY_AMT" + str(x)]) 

        else linha["PAY_AMT" + str(x)], axis=1)
# Garante a reprodutibilidade dos resultados.

random.seed(101)



# Substitui os valores faltantes de LIMIT_BAL.

teste["LIMIT_BAL"].fillna(teste["LIMIT_BAL"].apply(lambda x: random.randrange(10000, 500001, 10000)), inplace=True)
teste.info()
teste.describe()
from sklearn.ensemble import GradientBoostingClassifier
X = treino.drop("default.payment.next.month", axis=1) # Features.

y = treino["default.payment.next.month"]              # Target.
gbc = GradientBoostingClassifier()

gbc.fit(X, y) # Treina o modelo.
y_pred = gbc.predict(teste) # Predição do target.
saida = pd.DataFrame({"ID": teste_ID["ID"], "default.payment.next.month": y_pred}) # Cria o dataframe de saída (i.e., a ser exportado).

saida.to_csv("1.csv", index=False) # Exporta o dataframe criado no passo anterior.
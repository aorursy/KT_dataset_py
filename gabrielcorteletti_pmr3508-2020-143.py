import pandas as pd

import numpy as np

import sklearn as sk

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



%matplotlib inline
#Dados de treino:

adult = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=["Id"],na_values = "?")



#Dados de teste:

testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col = ["Id"], na_values = "?")
adult.head()
adult.info()
adult.describe(exclude=[np.number])
adult.describe(include=[np.number])
sns.pairplot(adult, vars=["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", 

                          "hours.per.week"], hue="income", diag_kws={'bw':"1.0"}, corner=True)

plt.show()
#copia do dataset transformando os valores de income em numericos (0 ou 1) para permitir a utilizacao do heat map:

adult_copy = adult.copy()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

adult_copy["income"] = le.fit_transform(adult_copy['income'])



#heat map:

plt.figure(figsize=(10,10))

mask = np.zeros_like(adult_copy.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(adult_copy.corr(), square=True, vmin=-1, vmax=1, annot = True, linewidths=.5, mask=mask)

plt.show()
plt.figure(figsize=(5,5))

sns.boxplot(data=adult, x="income", y="age")

plt.show()
plt.figure(figsize=(5,5))

sns.boxplot(data=adult, x="income", y="education.num")

plt.show()
plt.figure(figsize=(5,5))

sns.boxplot(data=adult, x="income", y="capital.gain")

plt.show()
plt.show(sns.scatterplot(data=adult, x="income",y="age"))

plt.show(sns.scatterplot(data=adult, x="income",y="education.num"))

plt.show(sns.scatterplot(data=adult, x="income", y="capital.gain"))
plt.figure(figsize=(5,5))

sns.boxplot(data=adult,x="income",y="capital.loss")

plt.show()
plt.figure(figsize=(5,5))

sns.scatterplot(data=adult, x="income", y="capital.loss")

plt.show()
plt.figure(figsize=(5,5))

sns.boxplot(data=adult,x="income",y="hours.per.week")

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(data=adult_copy, x="income", y="workclass")

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(data=adult_copy, x="income", y="education")

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(data=adult_copy, x="income", y="marital.status")

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(data=adult_copy, x="income", y="occupation")

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(data=adult_copy, x="income", y="relationship")

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(data=adult_copy, x="income", y="race")

plt.show()
plt.figure(figsize=(5,5))

sns.barplot(data=adult_copy, x="income", y="sex")

plt.show()
plt.figure(figsize=(8,8))

sns.barplot(data=adult_copy, x="income", y="native.country")

plt.show()
adult["native.country"].value_counts().plot(kind="pie", figsize = (8,8))

plt.show()
adult = adult.drop(labels=["fnlwgt", "education", "native.country"], axis="columns")

adult.head()
adult.drop_duplicates(keep="first", inplace=True)
adult.isna().sum()
adult[["workclass", "occupation"]].describe()
adult["workclass"] = adult["workclass"].fillna("Private")

adult["occupation"] = adult["occupation"].fillna("Prof-specialty")
adult.isna().sum()
Y_train = adult.pop("income")
#separando conforme as classificações:

numerical = adult[["age", "education.num", "hours.per.week"]].copy()

sparse = adult[["capital.gain", "capital.loss"]].copy()

categorical = adult[["workclass", "marital.status", "occupation", "relationship", "race", "sex"]].copy()



#criando uma nova coluna index:

numerical.reset_index(drop=True, inplace=True)

sparse.reset_index(drop=True, inplace=True)

categorical.reset_index(drop=True, inplace=True)



#transformando os dados categóricos em númericos:

categorical = pd.get_dummies(categorical)
categorical.head()
categorical.drop("sex_Female", axis="columns", inplace=True)
#função parar unir as variáveis anteriormente separadas por classificação:

def merge_tables(tables):

    new_table = pd.DataFrame()

    for cur_table in tables:               #para cada tipo

        for col in cur_table.columns:      #para cada coluna

            new_table[col] = cur_table[col]

    return new_table
X_train = merge_tables([numerical, sparse, categorical])

X_train.head()
best_score = 0

best_k = 0

for k in range(10,35):

    knn = KNeighborsClassifier(n_neighbors=k)

    cur_score = cross_val_score(knn, X_train, Y_train, cv=5, scoring="accuracy").mean()

    if cur_score > best_score:

        best_score = cur_score

        best_k = k

print("Melhor k:",best_k)

print("Melhor Acurácia:", best_score)
knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(X_train, Y_train)
#1)Removendo as colunas não selecionadas:

testAdult = testAdult.drop(labels=["fnlwgt", "education", "native.country"], axis="columns")





#2)Missing data:

testAdult["workclass"] = testAdult["workclass"].fillna("Private")

testAdult["occupation"] = testAdult["occupation"].fillna("Prof-specialty")





#3)Transformando dados categóricos em númericos:

#3.1)separando conforme as classificações:

testNumerical = testAdult[["age", "education.num", "hours.per.week"]].copy()

testSparse = testAdult[["capital.gain", "capital.loss"]].copy()

testCategorical = testAdult[["workclass", "marital.status", "occupation", "relationship", "race", "sex"]].copy()



#3.2)criando uma nova coluna index:

testNumerical.reset_index(drop=True, inplace=True)

testSparse.reset_index(drop=True, inplace=True)

testCategorical.reset_index(drop=True, inplace=True)



#3.3)transformando os dados categóricos em númericos:

testCategorical = pd.get_dummies(testCategorical)



#3.4)removendo "sex_Female":

testCategorical.drop("sex_Female", axis="columns", inplace=True)



#3.5)unindo novamente:

X_test = merge_tables([testNumerical, testSparse, testCategorical])
X_test.head()
Y_test_Pred = knn.predict(X_test)

Y_test_Pred
submission = pd.DataFrame()



submission[0] = X_test.index

submission[1] = Y_test_Pred

submission.columns = ["Id", "income"]
submission.head()
submission.to_csv("submission.csv", index=False)
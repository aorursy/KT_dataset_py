#PMR3508 - Tarefa1 Adult
#hash: PMR3508-2018-d59e43f3c1
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
%matplotlib inline
print(os.listdir("../input/dataset-adult/"))
#Adult Data
adult = pd.read_csv("../input/dataset-adult/train_data.csv",sep=",", na_values="?")
adult.shape
adult.head(3)
adult.info()
adult.describe()
adult['native.country'].value_counts()
adult['race'].value_counts().plot(kind="pie")
#distribuição de idade
adult["age"].plot(kind='hist',bins=15);
#agrupando atributo 'income' e 'sex' para cada idade
df=adult.groupby(["income","sex"]).mean()
df['age'].plot(kind="bar")
#Proporção de sexo por 'income'
df2=adult.groupby(["income","sex"]).size().unstack().plot(kind='bar',stacked=False)
#proporção de sexo por ocupação!
df2=adult.groupby(["occupation","sex"])['race'].size().unstack().plot(kind='barh',stacked=True)
#drop colunas empty e index "Id"
na_adult=adult.set_index("Id").dropna()
test_adult= pd.read_csv("../input/dataset-adult/test_data.csv",sep=",",na_values="?")
test_adult=test_adult.set_index("Id")
#armazena todos os dados de treino e de teste (numericos e categoricos)
X_adult = na_adult.iloc[:,:-1]
Y_adult = na_adult.income
X_test = test_adult.iloc[:,:]
#treino e teste apenas de dados numericos adult
num_cols=["age","education.num","capital.gain","capital.loss","hours.per.week"]

X_num=X_adult[num_cols]
Y_num=Y_adult
X_test= X_test[num_cols]
#importacao de bibliotecas de ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
#learning e predict
knn=KNeighborsClassifier(n_neighbors=9) #instacia model
scores = cross_val_score(knn,X_num,Y_num,cv=10) #validacao cruzada
knn.fit(X_num,Y_num)
Y_testpredict=knn.predict(X_test)
scores
#observando tipo de dados..
test_adult.dtypes
#Converte 'object columns para 'str', pois object pode conter dados em outro formato:
convert_cols=['workclass','education','marital.status','occupation','race','relationship',
              'sex','native.country']
test_adult[convert_cols] = test_adult[convert_cols].astype(str)
#testes..
test_adult.columns
#treino e teste de dados numericos e categoricos:
Xencode_adult= na_adult.iloc[:,:-1].apply(LabelEncoder().fit_transform)
Xencode_test_adult = test_adult.apply(LabelEncoder().fit_transform)

X_adult = Xencode_adult
X_test = Xencode_test_adult
Yfit_adult= LabelEncoder().fit(na_adult["income"])
Y_adult = Yfit_adult.transform(na_adult["income"])
#learning e predict
knn =KNeighborsClassifier(n_neighbors=10)
scores = cross_val_score(knn,X_adult,Y_adult,cv=10)
knn.fit(X_adult,Y_adult)
scores
Ytest_predict= knn.predict(X_test)
print(Ytest_predict)
X_adult.columns
#escolha de atributos para melhor predict
atributos=atributos=["age","workclass","education.num","occupation","sex","marital.status","capital.gain","capital.loss"]
X_adult = Xencode_adult[atributos]
X_test = Xencode_test_adult[atributos]
#Escolhendo valor ótimo de k por validação cruzada:
k_range=list(range(1,35))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_adult, Y_adult, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
# valor k do kNN e relação a acurácia da validação cruzada:
plt.plot(k_range, k_scores)
plt.xlabel('k')
plt.ylabel('Acurácia da validação cruzada')
#Escolhendo k=27 p kNN
knn =KNeighborsClassifier(n_neighbors=27)
knn.fit(X_adult,Y_adult)
scores
Ytest_predict= knn.predict(X_test)
#dados de submissao
label_out = Yfit_adult.inverse_transform(Ytest_predict)
df_out = pd.DataFrame({'Id': X_test.index,'income':label_out})
df_out.to_csv('submission_adult.csv',index=False)
pd.read_csv("submission_adult.csv")

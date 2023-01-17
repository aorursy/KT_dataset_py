import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import time



from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier 

%matplotlib inline

data_treino="../input/adult-pmr3508/train_data.csv"

data_teste="../input/adult-pmr3508/test_data.csv"
data_adult=pd.read_csv(data_treino, names=

        ["age", "workclass", "fnlwgt", "education", "education.num", "marital.status",

        "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week", "country", "target" ],

        

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

base_teste=pd.read_csv(data_teste, sep=r'\s*,\s*',

        engine='python',

        na_values="?")
data_adult.shape
data_adult
n_dadult= data_adult.dropna()

n_badult= base_teste.dropna()
n_dadult #base de treino sem as linhas com dados faltantes
n_badult # base de testes sem as linhas com dados faltantes 
n_dadult["sex"].value_counts().plot(kind="pie") #análise de sexos
n_dadult["marital.status"].value_counts().plot(kind="bar") #análise de status civil
n_dadult["occupation"].value_counts().plot(kind="bar") #análise de profissões
# Transformando as variáveis categóricas em números (encoding)

def number_encode_features(df):

    result = df.copy()

    encoders = {}

    for column in result.columns:

        if result.dtypes[column] == np.object:

            encoders[column] = preprocessing.LabelEncoder()

            result[column] = encoders[column].fit_transform(result[column])

    return result, encoders



# Calcular a correlação e plotar um HeatMap

encoded_data, _ = number_encode_features(n_dadult)

sns.heatmap(encoded_data.corr(), square=True,vmin=-1, vmax=1)

plt.show()
sns.stripplot('education.num', 'target', data=encoded_data) 
n_dadult[["education", "education.num"]].head(15)
#como a coluna education_num é a que melhor caracteriza a educação, podemos fazer:

del n_dadult["education"]
#selecionaremos alguns atributos de acordo com as observações acima

attributes = ["sex", "workclass", "education_num", "occupation", "marital_status"]
n_dadult
encoded_data
n_badult
attributes = ["sex", "workclass", "education.num", "occupation", "marital.status"]
train_adult_x = encoded_data[attributes]

train_adult_y = encoded_data.target
test_adult_x = n_badult.apply(preprocessing.LabelEncoder().fit_transform) #já numerizando as variáveis categóricas
# utilizaremos os dados numéricos para a implementação dos 3 algorítimos

Xtreino=encoded_data[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]

Ytreino=encoded_data[["target"]]

Xteste =test_adult_x[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
# fazendo a regressão logistica

logreg = LogisticRegression()

logreg.fit(Xtreino,Ytreino)

Ypred=logreg.predict(Xteste)

# avaliação por validacao cruzada

scores = cross_val_score(logreg, Xtreino, Ytreino, cv=10)

med_logreg=np.mean(scores) #media dos valores da validação cruzada

# avaliação da acurácia

Ypredt=logreg.predict(Xtreino)

ac_logreg=accuracy_score(Ytreino,Ypredt,normalize=True,sample_weight=None)
ac_logreg
med_logreg
# fazendo uma floresta aleatória

rand=RandomForestClassifier(n_estimators=100) #utilizando uma floresta de 100 árvores

rand.fit(Xtreino,Ytreino)

Ypred=rand.predict(Xteste)

# avaliação por validacao cruzada

scores = cross_val_score(rand, Xtreino, Ytreino, cv=10)

med_randflor=np.mean(scores) #media dos valores da validação cruzada

# avaliação da acurácia

Ypredt=rand.predict(Xtreino)

ac_randflor=accuracy_score(Ytreino,Ypredt,normalize=True,sample_weight=None)
ac_randflor #acurácia
med_randflor
arvore = DecisionTreeClassifier()

arvore.fit(Xtreino,Ytreino)

Ypred=arvore.predict(Xteste)

# avaliação por validacao cruzada

scores = cross_val_score(arvore, Xtreino, Ytreino, cv=10)

med_arvore=np.mean(scores) #media dos valores da validação cruzada

# avaliação da acurácia

Ypredt=arvore.predict(Xtreino)

ac_arvore=accuracy_score(Ytreino,Ypredt,normalize=True,sample_weight=None)
ac_arvore
med_arvore
knn = KNeighborsClassifier(n_neighbors = 23, p = 1)

start = time.time()

scores = cross_val_score(knn, Xtreino, Ytreino, cv = 10)

print('K-Nearest Neighbors CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(scores.mean(), scores.std()))

print ('Time elapsed: {0:1.2f}\n'.format(time.time()-start))
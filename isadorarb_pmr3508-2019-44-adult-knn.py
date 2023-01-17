import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing
teste = pd.read_csv("../input/adult-pmr3508/test_data.csv")

treino = pd.read_csv("../input/adult-pmr3508/train_data.csv")

nteste = teste.dropna()

ntreino = treino.dropna()
ntreino["income"] = ntreino["income"].map({"<=50K": 0, ">50K":1})

ntreino["sex"] = ntreino["sex"].map({"Male": 0, "Female":1})
ntreino.head()
sns.heatmap(ntreino.corr(), annot=True, vmin=-1, vmax=1)
sns.lineplot('education.num', 'income', data=ntreino)
sns.lineplot('sex', 'income', data=ntreino)
sns.lineplot('hours.per.week', 'income', data=ntreino)
sns.pairplot(ntreino, hue='income')
names = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']

# Get column names first

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

x = ntreino.loc[:, names].values

x = scaler.fit_transform(x)

scaled_df = pd.DataFrame(x, columns=names)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(scaled_df)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])
data_treino_principal = pd.concat([principalDf, ntreino[['income']]], axis = 1)
sns.pairplot(data_treino_principal, hue='income')
y = np.array(ntreino['income'])

num_ntreino = ntreino.apply(preprocessing.LabelEncoder().fit_transform)

x = np.array(num_ntreino[["age", "workclass", "education.num", 

        "occupation", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week"]]) #foram consideradas as categorias mais importantes

better=0

melhork=0



for c in range(1,101):

    knn = KNeighborsClassifier(n_neighbors=c)

    scores = cross_val_score(knn, x, y, cv=10)

    if(scores.mean() > better):

        melhork = c

        better = scores.mean() 
melhork #K=67
better #A melhor acurácia
knn = KNeighborsClassifier(n_neighbors=melhork)

knn.fit(x,y)
nteste["sex"] = nteste["sex"].map({"Male": 0, "Female":1})

num_nteste = nteste.apply(preprocessing.LabelEncoder().fit_transform)

x_nteste = np.array(num_nteste[["age", "workclass", "education.num", 

        "occupation", "race", "sex", "capital.gain", "capital.loss",

        "hours.per.week"]])

ynteste_predicao = knn.predict(x_nteste)
ynteste_predicao
df = pd.DataFrame(ynteste_predicao)

df = df.replace(0,"<=50K")

df = df.replace(1,">50K")
df
arquivo = "predicao.csv"

predicao = pd.DataFrame(num_nteste, columns = ["income"])

predicao["income"] = df

predicao.to_csv(arquivo, index_label="Id")

predicao
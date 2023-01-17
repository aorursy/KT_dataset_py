import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
treino = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = "?")

teste = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values = "?")
treino.head()
treino.shape

teste.head()
treino.shape
treino.info()
treino.isnull().sum()
treinoNA = treino.dropna()

testeNA = teste.dropna()
from sklearn import preprocessing



numtreino = treinoNA.apply(preprocessing.LabelEncoder().fit_transform)

numteste = testeNA.apply(preprocessing.LabelEncoder().fit_transform)



numtreino.head()
numtreino.hist(bins=50, figsize=(20, 20))


corr = treinoNA.apply(preprocessing.LabelEncoder().fit_transform).corr()

plt.figure(figsize=(15,15))



sns.heatmap(corr, annot=True)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



Xadult = numtreino.loc[:, ('age', 'education', 'marital.status',

                        'relationship', 'sex', 'capital.gain',

                        'capital.loss', 'hours.per.week')]

Yadult = numtreino.loc[:, ('income')]



XtestAdult = numteste.loc[:, ('age', 'education', 'marital.status',

                                'relationship', 'sex', 'capital.gain',

                                'capital.loss', 'hours.per.week')]

knn = KNeighborsClassifier(n_neighbors=28)

knn.fit(Xadult,Yadult)



Pred = knn.predict(XtestAdult)





result = pd.DataFrame({'income': Pred})

result.to_csv("submission.csv", index=True, index_label='Id')

result

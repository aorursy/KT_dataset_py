import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
arquivo1 = '../input/test.csv'
tester = pd.read_csv(arquivo1, engine = 'python')
tester
arquivo2 = '../input/train.csv'
trainer = pd.read_csv(arquivo2, engine = 'python')
trainer.shape
trainer.head()
tester.head()
trainer['v2a1'].head()
#Essa função conta o número de ocorrências de zeros e valores vazios numa dada coluna
def Count_Zs_Ns(column):
    zeros1 = 0
    zeros2 = 0
    nans1 = 0
    nans2 = 0
    for i in trainer[column]:
        if i == 0:
            zeros2+=1
        if str(i) == 'nan':
            nans2+=1
    for i in tester[column]:
        if i == 0:
            zeros1+=1
        if str(i) == 'nan':
            nans1+=1
    print('{} tem {} zeros e {} NaNs na base de treino e {} zeros e {} NaNs na base de testes'.format(column,zeros2,nans2,zeros1,nans1))
Count_Zs_Ns('v2a1')
Count_Zs_Ns('v18q1')
#Preenchendo os vazios das colunas com 0s
trainer['v18q1'] = trainer['v18q1'].fillna(0)
trainer['v2a1'] = trainer['v2a1'].fillna(0)
tester['v18q1'] = tester['v18q1'].fillna(0)
tester['v2a1'] = tester['v2a1'].fillna(0)
trainer_filled = trainer.fillna(-1)
tester_filled = tester.fillna(-1)
trainer_filled['idhogar'].head()
trainer_filled
trainer_filled = trainer_filled.replace('no',0)
trainer_filled = trainer_filled.replace('yes',1)
tester_filled = tester_filled.replace('no',0)
tester_filled = tester_filled.replace('yes',1)
#Verificando possíveis relações de dados com a variável de interesse
pd.crosstab(trainer_filled['SQBescolari'],trainer_filled['Target'])
pd.crosstab(trainer_filled['SQBhogar_total'],trainer_filled['Target'])
trainer_filled['refrig'].value_counts().plot(kind='pie')
trainer_filled['escolari'].value_counts().plot(kind='bar')
Ytrainer = trainer_filled.Target
Xtrainer = trainer_filled.drop(['Id','idhogar','Target'], axis = 1)
Xtester = tester_filled.drop(['Id','idhogar'], axis = 1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=175)
from sklearn.model_selection import cross_val_score as cross
scores = cross(knn, Xtrainer, Ytrainer, cv=10)
scores
scores.sum()/10
knn.fit(Xtrainer,Ytrainer)
Ytrainerpred = knn.predict(Xtrainer)
from sklearn.metrics import accuracy_score as acs
acs(Ytrainer,Ytrainerpred)
Predict = knn.predict(Xtester)
Predict
Id = tester['Id']
Submit = pd.DataFrame()
Submit.insert(0,'Id',Id)
Submit.insert(1,'Target',Predict)
file = open('sample_submission.csv','w')
file.write(pd.DataFrame.to_csv(Submit, index=False))
file.close()

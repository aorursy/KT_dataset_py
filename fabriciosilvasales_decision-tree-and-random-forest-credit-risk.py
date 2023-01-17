import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
data = pd.read_csv("/kaggle/input/data-credit/credit_data.csv")

data
m = np.mean(data['age'])



data.loc[data.age < 0, 'age'] = m

data.loc[data.age.isna() , 'age'] = m
previsores = data.iloc[:,1:4].values

classe = data.iloc[:, 4].values
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

previsores = scaler.fit_transform(previsores)
from sklearn.model_selection import train_test_split



previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)
from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

tree.fit(previsores_treinamento, classe_treinamento)

previsoes_tree = tree.predict(previsores_teste)
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix

import matplotlib.pyplot as plt



plot_confusion_matrix(tree, previsores_teste, classe_teste,display_labels=["Mau Pagador","Bom Pagador"])

precisao = accuracy_score(classe_teste, previsoes_tree)



print("Acurácea do Modelo: "+str(precisao*100)+"%")
from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=35, criterion='entropy', random_state=0)

forest.fit(previsores_treinamento, classe_treinamento)

previsoes_forest = forest.predict(previsores_teste)
plot_confusion_matrix(forest, previsores_teste, classe_teste,display_labels=["Mau Pagador","Bom Pagador"])

precisao = accuracy_score(classe_teste, previsoes_forest)



print("Acurácea do Modelo: "+str(precisao*100)+"%")
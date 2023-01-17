# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/dados-de-simulao-de-sep/CB_166_1.txt', sep = ',', header = None, engine = 'python')

df = pd.DataFrame(data = df)

df
#bibliotecas necessárias

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

#biblioteca para "conversão" de variaveis categoricas (strings) em números (integer)

from sklearn.preprocessing import LabelEncoder

#biblioteca para criar tabela confusão e medir precisão

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier
previsores = df.iloc[:, 0:1114].values 

previsores
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

previsores = minmax.fit_transform(previsores)

previsores
classes = df.iloc[:, 1114].values

classes
LEncoder = LabelEncoder()  #função label encoder é o "etiquetador" das categorias

classes = LEncoder.fit_transform(classes)

classes 
x_train, x_test, y_train, y_test = train_test_split(previsores, classes, test_size = 0.3, random_state = 0)
naiveb = GaussianNB()

naiveb.fit(x_train, y_train)
predicao = naiveb.predict(x_test)

predicao
#comparando os dados previstos com dados anteriores por matriz de confusão

confusion = confusion_matrix(predicao, y_test)

confusion
accu = accuracy_score(predicao, y_test)

accu
from yellowbrick.classifier import ConfusionMatrix
viz = ConfusionMatrix(GaussianNB())

viz.fit(x_train, y_train)

viz.score(x_test, y_test)

viz.poof()
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

tree = DecisionTreeClassifier()

tree.fit(x_train, y_train)
export_graphviz(tree, out_file ='tree.dot')
previsao = tree.predict(x_test)

previsao
conf = confusion_matrix(y_test, previsao)

conf
acerto = accuracy_score(y_test, previsao)

acerto
from sklearn.svm import SVC
svm = SVC()

svm.fit(x_train, y_train)

prev = svm.predict(x_test)

acerto_2 = accuracy_score(y_test, prev)

acerto_2
forest = ExtraTreesClassifier()

forest.fit(x_train, y_train)

importance = forest.feature_importances_

importance
forest = RandomForestClassifier(n_estimators = 100)

forest
forest.fit(x_train, y_train)
predicties = forest.predict(x_test)

predicties
confusion = confusion_matrix(predicties, y_test)

confusion
accuracy_score(predicties, y_test)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

import graphviz

from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential

from keras.layers.core import Dense



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/heart-disease-uci/heart.csv")

data.describe()
fig, axs = plt.subplots(2, 3, figsize = (15,10))

axs[0, 0].hist(data["age"], bins=60, alpha=1, edgecolor = 'black',  linewidth=1)

axs[0, 0].set_title('Edades')

axs[0, 1].hist(data["trestbps"], bins=60, alpha=1, edgecolor = 'black',  linewidth=1)

axs[0, 1].set_title('Presión en reposo')

axs[0, 2].hist(data["chol"], bins=60, alpha=1, edgecolor = 'black',  linewidth=1)

axs[0, 2].set_title('Colesterol serico')

axs[1, 0].hist(data["thalach"], bins=60, alpha=1, edgecolor = 'black',  linewidth=1)

axs[1, 0].set_title('Frecuencia cardiaca max')

axs[1, 1].hist(data["oldpeak"], bins=60, alpha=1, edgecolor = 'black',  linewidth=1)

axs[1, 1].set_title('Depresión inducida por ejercicio')



plt.show()
labels1 ='Hombre','Mujer'

labels2 = '0','2','1','3'

labels3 = 'Si','No'

labels4 = '1','0','2'

labels5 = 'No','Si'

labels6 = '2','1','0'

labels7 = '0','1','2','3','4'

labels8 = '2','3','1','0'

labels9 = 'Si','No'



fig, axs = plt.subplots(3, 3, figsize = (10,10))

axs[0, 0].pie(data["sex"].value_counts(),labels=labels1, autopct='%1.1f%%', shadow=True)

axs[0, 0].set_title('Sexo')

axs[0, 1].pie(data["cp"].value_counts(),labels=labels2, autopct='%1.1f%%', shadow=True)

axs[0, 1].set_title('Tipo de dolor de pecho')

axs[0, 2].pie(data["fbs"].value_counts(),labels=labels3, autopct='%1.1f%%', shadow=True)

axs[0, 2].set_title('Glucosa > 120')

axs[1, 0].pie(data["restecg"].value_counts(),labels=labels4, autopct='%1.1f%%', shadow=True)

axs[1, 0].set_title('Electro en reposo')

axs[1, 1].pie(data["exang"].value_counts(),labels=labels5, autopct='%1.1f%%', shadow=True)

axs[1, 1].set_title('Angina por ejercicio')

axs[1, 2].pie(data["slope"].value_counts(),labels=labels6, autopct='%1.1f%%', shadow=True)

axs[1, 2].set_title('Pendiente del segmento de pico')

axs[2, 0].pie(data["ca"].value_counts(),labels=labels7, autopct='%1.1f%%', shadow=True)

axs[2, 0].set_title('Vasos ppals pintados por flourosopía')

axs[2, 1].pie(data["thal"].value_counts(),labels=labels8, autopct='%1.1f%%', shadow=True)

axs[2, 1].set_title('Thal')

axs[2, 2].pie(data["target"].value_counts(),labels=labels9, autopct='%1.1f%%', shadow=True)

axs[2, 2].set_title('Ataque al corazon')



plt.show()

atributos = data[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]].values

target = data["target"].values



X_entrenamiento,X_test,y_entrenamiento,y_test = train_test_split(atributos,target,test_size=0.2)
arbol1 = DecisionTreeClassifier()

arbol1.fit(X_entrenamiento,y_entrenamiento)



print("Score del entrenamiento")

print(arbol1.score(X_entrenamiento,y_entrenamiento))

print("Score del test")

print(arbol1.score(X_test,y_test))
features=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

export_graphviz(arbol1,out_file='arbol1.dot',class_names='target', feature_names=features, impurity=False, filled=True)

with open('arbol1.dot') as f:

    dot_graph=f.read()

graphviz.Source(dot_graph)
carac = data.shape[1]



for i in range(13):

    print(features[i],": ",arbol1.feature_importances_[i]*100)
arbol2 = DecisionTreeClassifier(criterion= "entropy",max_depth=6, min_samples_split = 15)

arbol2.fit(X_entrenamiento,y_entrenamiento)



print("Score del entrenamiento")

print(arbol2.score(X_entrenamiento,y_entrenamiento))

print("Score del test")

print(arbol2.score(X_test,y_test))
export_graphviz(arbol2,out_file='arbol2.dot',class_names='target', feature_names=features, impurity=False, filled=True)



with open('arbol2.dot') as f:

    dot_graph=f.read()

graphviz.Source(dot_graph)
carac = data.shape[1]

for i in range(13):

    print(features[i],": ",arbol2.feature_importances_[i]*100)

print("Matriz de confusión")

print(confusion_matrix(arbol2.predict(X_test), y_test))
bayes = GaussianNB()

bayes.fit(X_entrenamiento,y_entrenamiento)



print("Score del test")

print(bayes.score(X_test,y_test))

print("Score del entrenamiento")

print(bayes.score(X_entrenamiento,y_entrenamiento))
print("Matriz de confusión")

print(confusion_matrix(bayes.predict(X_test), y_test))
red = Sequential()

red.add(Dense(16, input_dim=13, activation='sigmoid'))

red.add(Dense(14, activation='relu'))

red.add(Dense(1, activation='sigmoid'))



red.compile(loss='mean_squared_error',

              optimizer='adam',

              metrics=['binary_accuracy'])



red.fit(X_entrenamiento, y_entrenamiento, epochs=1000)



scoreT = red.evaluate(X_test, y_test)

scoreE = red.evaluate(X_entrenamiento, y_entrenamiento)



print("Score del test")

print("%s: %.2f%%" % (red.metrics_names[1], scoreT[1]*100))

print("Score del entrenamiento")

print("%s: %.2f%%" % (red.metrics_names[1], scoreE[1]*100))

for layer in red.layers:

    g=layer.get_config()

    h=layer.get_weights()

    print ('Weights of layer: ',h)
print("Matriz de confusión")

print(confusion_matrix(red.predict_classes(X_test), y_test))
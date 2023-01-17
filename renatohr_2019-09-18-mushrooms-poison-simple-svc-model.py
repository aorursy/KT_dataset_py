# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# lendo o arquivo CSV (dataset)

df = pd.read_csv('../input/mushrooms.csv')



#jogando fora a coluna veil-type, porque é inutil para prever alguma coisa, já que possui 1 valor possível apenas

df.drop("veil-type", axis=1, inplace=True) #useless --> only 1 possible value



#printando 5 primeiras linhas do dataset, para visualizar

df.head()
#machine learning só trabalha com numeros, então precisamos converter as strings em numeros.

#converting strings to numbers

#would be better to check if there is any crescent-like feature, instead of converting randomly to numbers, but i'm too lazy

from sklearn import preprocessing



for column in df.columns:

    values = df[column].values

    le = preprocessing.LabelEncoder()

    le.fit(values)

    # print(le.classes_)

    df[column] = le.transform(values)



#esse é o dataframe com apenas numeros agora

df.head()
#obs: class = se o cogumelo é poisonous ou não



#X são as Features --> todas colunas exceto class, ou seja, todos os atributos do cogumelo que permitem a gente prever a classe dele

X = df.drop("class",axis=1).values



#y é o label --> apenas a coluna class, ou seja, é o resultado que queremos prever com base nas features

y = df["class"].values
from sklearn.model_selection import train_test_split



#separando o dataset em 80% treino do machine learning e 20% teste posterior, para evitar overfit

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(len(X))

print(len(X_train))

print(len(X_test))
from sklearn import svm



#vamos usar o machine learning tipo SVC

clf = svm.SVC(gamma='scale')

clf.fit(X_train, y_train)
#checando sucesso/precisão desse modelo com função do sklearn

from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_test, y_test,cv=5)

for score in scores:

    print(score)
#print(X_test[200])

#print(y_test[200])

#clf.predict([X_test[200]])



wons = 0

losses = 0



#checando sucesso/precisão desse modelo com método "manual"

for i in range(0,len(X_test)):

    prediction = clf.predict([X_test[i]])[0]

    right = y_test[i]

    if prediction == right:

        answer = 'OK'

        wons += 1

    else:

        answer = 'WRONG'

        losses += 1

    print(f'Test {i} {answer}: ML model predicted {prediction}. The right is {right}')

print(f'Wons: {wons}, Losses: {losses}, Accuracy: {wons/(wons+losses)}')
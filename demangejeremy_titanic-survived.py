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
# Importation des librairies

from sklearn.neighbors import KNeighborsClassifier
# Importation des données

train_x = pd.read_csv('/kaggle/input/titanic/train.csv')
# Prétraitements des données

train_x = train_x[['Pclass', 'Sex', 'Age', 'Survived']]

train_x.dropna(axis=0, inplace=True)

train_x['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

train_y = train_x[['Survived']]

train_x = train_x[['Pclass', 'Sex', 'Age']]



# Afficher les données

train_x.head()
# Initialisation du model

model = KNeighborsClassifier(n_neighbors=3)
# Entrainer le model

model.fit(train_x, train_y.values.ravel())
# Afficher le score

model.score(train_x, train_y.values.ravel())
def survived(model, pclass=3, sex=0, age=20):

    x = np.array([pclass, sex, age]).reshape(1, 3)

    result = model.predict(x)

    if result == 0:

        print("X - Vous n'avez pas survécu au drame.")

    else:

        print("O - Vous avez survécu au drame du Titanic.")

    

survived(model, 3, 1, 20)
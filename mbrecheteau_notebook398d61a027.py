# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



Data_path = "../input/"



# Any results you write to the current directory are saved as output.
### Exploration du jeu d'entrainement
train = pd.read_csv(Data_path + 'train.csv')
train.describe()
print("Nombre de label distinct : ",train.label.nunique())

print("Nombre d'enregistrement dans le jeu d'entrainement (ligne, colonne) : ", train.shape)
train.label.values()
### Graphique données
plt.hist(train.label, 50, normed=1, facecolor='g', alpha=0.75)

plt.xlabel('Label')

plt.ylabel('Nombre de label')

plt.title('Représentation des labels')
### Verification de données
train.columns
train.label[1].count()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk # scikit-learn for the win



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/dcod19/train.csv")

test = pd.read_csv("/kaggle/input/dcod19/test.csv")

print("Chargement des fichiers train.csv et test.csv réussis sans erreur")



#from sklearn.impute import SimpleImputer

#imp = SimpleImputer(strategy="constant", fill_value = 0)

#filled_trainer = imp.fit_transform(train)

#print("Remplissage des valeurs indisponibles réussi")



y = train.loc[:,"Class"]

filled_trainer = filled_trainer[:, 1:190]



from sklearn import svm

clf = sk.svm.SVC()

print("Création du classifieur réussie")





clf.fit(filled_trainer, y)

clf





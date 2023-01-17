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
import os

import numpy as np

import pandas as pd

from os import listdir



#Les données sont centrées par hypothèse



data =pd.read_csv('/kaggle/input/decathlon.txt', sep="\t")



#On élimine les données inutiles qui ne sont pas nécessaire. 



newdata = data.drop(['Points', 'Rank', 'Competition'], axis=1)



X = newdata.values



newdata.head()
#Maintenant on va centrée reduire (stanardisées) les données.
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(X)

X_scaled = std_scale.transform(X)
from sklearn import decomposition





#Cela nous les 4 composantes les plus pertinantes.

PrinCompAnaly = decomposition.PCA(n_components=4)

PrinCompAnaly.fit(X_scaled)
print(PrinCompAnaly.explained_variance_ratio_)

print(PrinCompAnaly.explained_variance_ratio_.sum())



# La premières composantes explique 32% de la variance observée dans les données, la seconde 17%, la troisième 14% et la quatrième 10%.

#À eux quatre il explique 74% de la variance totale.
import matplotlib.pyplot as plt



# On projete X sur les composante principales



projectionX = PrinCompAnaly.transform(X_scaled)



# afficher chaque observation



plt.scatter(projectionX[:, 0], projectionX[:, 1],

    # colorer en utilisant la variable 'Rank'

    c=data.get('Rank'))



plt.xlim([-5.5, 5.5])

plt.ylim([-4, 4])

plt.colorbar()
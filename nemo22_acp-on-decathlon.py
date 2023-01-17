# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# charger les données

data = pd.read_csv('../input/decathlon.txt', sep="\t")

# éliminer les colonnes que nous n'utiliserons pas

my_data = data.drop(['Points', 'Rank', 'Competition'], axis=1)



# transformer les données en array numpy

X = my_data.values
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(X)

X_scaled = std_scale.transform(X)
X_scaled
from sklearn import decomposition



pca = decomposition.PCA(n_components=2)

pca.fit(X_scaled)
print (pca.explained_variance_ratio_)

print (pca.explained_variance_ratio_.sum())
import matplotlib.pyplot as plt

# projeter X sur les composantes principales

X_projected = pca.transform(X_scaled)



# afficher chaque observation

plt.scatter(X_projected[:, 0], X_projected[:, 1],

    # colorer en utilisant la variable 'Rank'

    c=data.get('Rank'))



plt.xlim([-5.5, 5.5])

plt.ylim([-4, 4])

plt.colorbar()
pcs = pca.components_



for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):

    # Afficher un segment de l'origine au point (x, y)

    plt.plot([0, x], [0, y], color='k')

    # Afficher le nom (data.columns[i]) de la performance

    plt.text(x, y, data.columns[i], fontsize='14')



# Afficher une ligne horizontale y=0

plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')



# Afficher une ligne verticale x=0

plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')



plt.xlim([-0.7, 0.7])

plt.ylim([-0.7, 0.7])
%matplotlib inline

import numpy as np 

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')
numFeatures = []

catFeatures = []



for col, val in train.iloc[0,:].iteritems():

    if type(val) is not str:

        numFeatures.append(col)

    elif type(val) is str:

        catFeatures.append(col)

        

# Remove id and loss from the numFeatures

numFeatures.remove('id')

numFeatures.remove('loss')

        

print(len(numFeatures), 'Numerical Features:', numFeatures, "\n")

print(len(catFeatures), 'Categorical Features:', catFeatures)
sc = StandardScaler()

train_nums_std = sc.fit_transform(train[numFeatures])
pca = PCA(n_components=None)

train_nums_pca = pca.fit_transform(train_nums_std)

varExp = pca.explained_variance_ratio_
cumVarExplained = []

nb_components = []

counter = 1

for i in varExp:

    cumVarExplained.append(varExp[0:counter].sum())

    nb_components.append(counter)

    counter += 1



plt.subplots(figsize=(8, 6))

plt.plot(nb_components, cumVarExplained, 'bo-')

plt.ylabel('Cumulative Explained Variance')

plt.xlabel('Number of Components')

plt.ylim([0.0, 1.1])

plt.xticks(np.arange(1, len(nb_components), 1.0))

plt.yticks(np.arange(0.0, 1.1, 0.10))
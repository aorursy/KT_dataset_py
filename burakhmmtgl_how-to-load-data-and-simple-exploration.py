import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Load one of the data files

with open("../input/pubChem_p_00025001_00050000.json") as f:

    data = json.load(f)
# Read energies and shape multipoles

energies = np.zeros(len(data))

multipoles = []

for i, molecule in enumerate(data):

    energies[i] = molecule['En']

    multipoles.append(molecule['shapeM'])



# Shape multipoles

multP = np.array(multipoles)
# Plot distribution of energies

fig = plt.hist(energies, bins=50)

plt.xlabel('Energy')

plt.ylabel('Counts')

plt.title('Distribution of Energies')

plt.show()
# Get principal components of multipoles and explore their correlation to energy

from sklearn.decomposition import PCA



pca = PCA(n_components=2) # Get the first 2

X_pca = pca.fit_transform(multP) 



fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))



ax1.scatter(X_pca[:,0], energies, color = 'red')

ax2.scatter(X_pca[:,1], energies, color = 'blue')



ax1.set_xlabel('PC-1')

ax2.set_xlabel('PC-2')

ax1.set_ylabel('Energy')

ax2.set_ylabel('Energy')

plt.suptitle("PC of Shape multipoles and Energy")

plt.show()
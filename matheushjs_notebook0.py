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
data = pd.read_csv("../input/Iris.csv", index_col='Id')

data
data.columns = np.array(['sl', 'sw', 'pl', 'pw', 'species'])

data.columns
species_dict = {}

species_dict_inv = dict(enumerate(data.species.unique()))

for i,j in species_dict_inv.items():

    species_dict[j] = i

species_dict
data.species = data.species.map(species_dict)

data
test_indexes = np.random.rand(data.shape[0]) >= 0.75

train_indexes = ~test_indexes



train_data = data[train_indexes]

test_data = data[test_indexes]
from sklearn.tree import DecisionTreeClassifier



col = ['sl', 'sw', 'pl', 'pw']



clf = DecisionTreeClassifier()

clf.fit(train_data[col].values, train_data['species'].values)



pred = clf.predict(test_data[col])

s = np.sum(pred == test_data['species'])

s/test_data.shape[0]
import matplotlib.pyplot as plt
data2 = data.copy()

data2.species = data2.species.map(species_dict_inv)

data2
colors = ['red', 'green', 'blue']



plt.figure(figsize=(12,6))

for i, c in zip(data2.species.unique(), colors):

    d = data2[data2.species == i]

    plt.scatter(d.sl, d.sw, color=c, label=i)



plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')

plt.legend()

plt.show()
colors = ['red', 'green', 'blue']



plt.figure(figsize=(12,6))

for i, c in zip(data2.species.unique(), colors):

    d = data2[data2.species == i]

    plt.scatter(d.pl, d.pw, color=c, label=i)

    

plt.xlabel('Petal Length')

plt.ylabel('Petal Width')

plt.legend()

plt.show()
col = ['sl', 'pl', 'pw']



clf2 = DecisionTreeClassifier()

clf2.fit(train_data[col].values, train_data['species'].values)



pred = clf2.predict(test_data[col])

s = np.sum(pred == test_data['species'])

s/test_data.shape[0]
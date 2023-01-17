import os

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
data = pd.read_csv('../input/oakland-street-trees.csv')
print('Rows: ', data.shape[0])

print('Columns: ', data.shape[1])



print('Columns Names: ')    

print([name for name in data.columns])
data.head(5)
data = data.drop(['STNAME','Location 1','OBJECTID', 'LOWWELL'], axis=1)

data.head(5)
data.info()
data.describe()
data.groupby('SPECIES').size().sort_values(ascending=False)
species = data.groupby('SPECIES').size()

print('Num of Species: ', len(species))
plt.figure(figsize=(8,5))

species.nlargest(15).plot.bar()
data = data[~data['SPECIES'].isin(['TBD', 'Other', 'Unknown'])]

print('Num of Species: ', len(data.groupby('SPECIES').size()))
species_set = ['Platanus acerifolia', 'Liquidambar styraciflua']



small_data = data[data['SPECIES'].isin(species_set)]
small_data.head(5)
X = small_data.drop('SPECIES', axis=1)

y = small_data['SPECIES']
clean_y = {'Platanus acerifolia':0, 'Liquidambar styraciflua':1}

y.replace(clean_y, inplace=True)
y.head(5)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=42)
print('TRAIN: \n x_train shape: ', x_train.shape)

print(' y_train shape: ', y_train.shape)

print('\nTEST: \n x_test shape: ', x_test.shape)

print(' y_test shape: ', y_test.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rf = RandomForestClassifier(n_estimators=100)



rf.fit(x_train, y_train)



train_accuracy = rf.score(x_train, y_train)

print('Train Acc: ', round(train_accuracy, 1))



test_accuracy = rf.score(x_test, y_test)

print('Test Acc: ', round(test_accuracy, 1))
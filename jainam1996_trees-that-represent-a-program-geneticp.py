# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.info()
data.head(5)
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

data.info()
diagnosis_all = list(data.shape)[0]

diagnosis_categories = list(data['diagnosis'].value_counts())



print("The data has {} diagnosis, {} malignant and {} benign.".format(diagnosis_all,

                                                                      diagnosis_categories[0], 

                                                                      diagnosis_categories[1]))
features_mean= list(data.columns[1:11])
plt.figure(figsize=(9,9))

sns.heatmap(data[features_mean].corr(), annot=True, square = True,cmap='coolwarm')

plt.show()
bins = 12

plt.figure(figsize=(9,9))

for i, feature in enumerate(features_mean):

    rows = int(len(features_mean)/2)

    

    plt.subplot(rows, 2, i+1)

    

    sns.distplot(data[data['diagnosis']=='M'][feature], bins=bins, color='red', label='M');

    sns.distplot(data[data['diagnosis']=='B'][feature], bins=bins, color='blue', label='B');

    

    plt.legend(loc='upper right')



plt.tight_layout()

plt.show()
data_shuffle=data.iloc[np.random.permutation(len(data))]

data_y=data_shuffle.reset_index(drop = True)
data.head(5)
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score



import time

data_y['diagnosis'] =data_y['diagnosis'].map({'M':1, 'B':0})

data_class=data_y['diagnosis'].values
training_indices,validation_indices = training_indices, testing_indices = train_test_split(data_y.index, 

                                                                                          stratify= data_class, train_size = 0.75, test_size = 0.25)
!pip install genepy
from operator import add, sub,mul

from genepy.core import EvolutionaryAlgorithm
def compute_fitness(tree, features, data):

    """

    Computes a normalized MAE on the predictions made by one tree.

    """

    predicted = [tree.predict(feat) for feat in features]

    difference = [abs(predicted[i] - data[i]) for i in range(len(data))]

    mae = reduce(lambda a,b: a+b, difference) / len(data)

    fitness = 1 / mae if mae != 0 else 1.0

    fitness /= len(tree.nodes)

    return fitness
parameters = {

  'min_depth':        3,

  'max_depth':        5,

  'nb_trees':         50,

  'max_const':        100,

  'func_ratio':       0.5,

  'var_ratio':        0.5,

  'crossover_prob':   0.8,

  'mutation_prob':    0.2,

  'iterations':       1000,

  'functions':        [add,sub,mul],

  'fitness_function': compute_fitness

}


from functools import reduce

ea = EvolutionaryAlgorithm(**parameters)

ea.fit(data_y.drop('diagnosis',axis=1).loc[training_indices].values,data_y.loc[training_indices,'diagnosis'].values)

predicted = ea.predict(testing_indices)
print('The Prediction accuracy is :', predicted* 0.1, '%')
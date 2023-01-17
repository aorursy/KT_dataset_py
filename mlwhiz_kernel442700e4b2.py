# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import random



def generator(max):

    number = 1

    while number < max:

        number += 1

        yield number



# Create as stream generator

stream = generator(10000)



# Doing Reservoir Sampling from the stream

k=5

reservoir = []

for i, element in enumerate(stream):

    if i+1<= k:

        reservoir.append(element)

    else:

        probability = k/(i+1)

        if random.random() < probability:

            # Select item in stream and remove one of the k items already selected

             reservoir[random.choice(range(0,k))] = element
reservoir
from sklearn.datasets import make_classification



X, y = make_classification(

    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],

    n_informative=3, n_redundant=1, flip_y=0,

    n_features=20, n_clusters_per_class=1,

    n_samples=100, random_state=10

)



data = pd.DataFrame(X)

data['target'] = y
num_0 = len(data[data['target']==0])

num_1 = len(data[data['target']==1])

print(num_0,num_1)



# random undersample



undersampled_data = pd.concat([ data[data['target']==0].sample(num_1) , data[data['target']==1] ])

print(len(undersampled_data))



# random oversample



oversampled_data = pd.concat([ data[data['target']==0] , data[data['target']==1].sample(num_0, replace=True) ])

print(len(oversampled_data))
from imblearn.under_sampling import TomekLinks



tl = TomekLinks(return_indices=True, ratio='majority')

X_tl, y_tl, id_tl = tl.fit_sample(X, y)
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_sample(X, y)
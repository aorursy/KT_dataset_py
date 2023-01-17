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
import sklearn

from sklearn.preprocessing import MinMaxScaler



import gplearn

from gplearn.genetic import SymbolicTransformer

from gplearn.functions import make_function
def get_data():

    train_df = pd.read_csv('../input/train.csv')

    test_df = pd.read_csv('../input/test.csv')

    

    train_data = train_df.values

    test_data = test_df.values



    train_features = np.float64(train_data[:, 2:])

    test_features = np.float64(test_data[:, 1:])

    

    # scaler = MinMaxScaler(feature_range=(-1, 1))

    # scaler.fit(np.concatenate([train_features, test_features], axis=0))

    # train_features = scaler.transform(train_features)

    # test_features = scaler.transform(test_features)

    

    train_target = np.float64(train_data[:, 1])

    

    test_ids = test_data[:, 0]

    

    return train_features, train_target, test_features, test_ids
train_features, train_target, test_features, test_ids = get_data()
# exp = make_function(function=lambda x: np.exp(x), name='exp', arity=1)

tanh = make_function(function=lambda x: (np.exp(2*x) - 1)/(np.exp(2*x) + 1), name='tanh', arity=1)
function_set = ['add', 'sub', 'mul', 'div',

                'sqrt', 'log', 'abs', 'neg', 'inv',

                'max', 'min', tanh]



gp = SymbolicTransformer(generations=750, population_size=2000,

                         hall_of_fame=100, n_components=50,

                         function_set=function_set,

                         parsimony_coefficient=0.0005,

                         max_samples=0.9, verbose=1,

                         random_state=0, n_jobs=3)



gp.fit(train_features, train_target)
train_genetic_features = gp.transform(train_features)

test_genetic_features = gp.transform(test_features)

np.save('train_gen_feat.npy', train_genetic_features)

np.save('test_gen_feat.npy', test_genetic_features)
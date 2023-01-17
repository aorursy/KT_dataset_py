# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import HuberRegressor

from sklearn import metrics

from sklearn.preprocessing import PolynomialFeatures



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
trainSet = pd.read_csv('../input/train.csv')

display(trainSet.head())
testSet = pd.read_csv('../input/test.csv')

display(testSet.head())
structures = pd.read_csv('../input/structures.csv')

display(structures.head())
# Map the atom structure data into train and test files



def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



trainSet = map_atom_info(trainSet, 0)

trainSet = map_atom_info(trainSet, 1)



testSet = map_atom_info(testSet, 0)

testSet = map_atom_info(testSet, 1)

display(trainSet.head())

display(testSet.head())
# https://www.kaggle.com/jazivxt/all-this-over-a-dog

# https://www.kaggle.com/artgor/molecular-properties-eda-and-models

train_p0 = trainSet[['x_0', 'y_0', 'z_0']].values

train_p1 = trainSet[['x_1', 'y_1', 'z_1']].values

test_p0 = testSet[['x_0', 'y_0', 'z_0']].values

test_p1 = testSet[['x_1', 'y_1', 'z_1']].values



trainSet['dist'] = np.linalg.norm(train_p0 - train_p1, axis=1)

testSet['dist'] = np.linalg.norm(test_p0 - test_p1, axis=1)



trainSet['dist_to_type_mean'] = trainSet['dist'] / trainSet.groupby('type')['dist'].transform('mean')

testSet['dist_to_type_mean'] = testSet['dist'] / testSet.groupby('type')['dist'].transform('mean')
# All atom_0 are hydrogens

assert all(trainSet["atom_0"].astype('category').cat.categories == ['H'])

assert all(testSet["atom_0"].astype('category').cat.categories == ['H'])
# atom_1 are carbon, hydrogen or nitrogen

print(trainSet["atom_1"].astype('category').cat.categories)

print(testSet["atom_1"].astype('category').cat.categories)
# We use the interaction types, that already include the type of atoms involved

print(testSet["type"].astype('category').cat.categories)

print(trainSet["type"].astype('category').cat.categories)
for i in trainSet["type"].astype('category').cat.categories.values:

    trainSet['type_'+str(i)] = (trainSet['type'] == i)

    testSet['type_'+str(i)] = (testSet['type'] == i)
model = HuberRegressor()
# Features to include (regressors)

regressors = ['type_1JHC', 'type_1JHN', 'type_2JHC', 'type_2JHH', 'type_2JHN', 

                                       'type_3JHC', 'type_3JHH', 'dist', 'dist_to_type_mean']
# Add bias, interaction term and quadratic and cubic terms

polyFeat = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)
trainX = polyFeat.fit_transform(np.array(trainSet[regressors]))
# Some features are uninformative:

# Interaction type features don't (statistically) interact as they are mutually exclusive

usefulFeatures = [i for i,x in enumerate(np.abs(np.sum(trainX, axis = 0))) if x > 0]

trainX = trainX[:,usefulFeatures]

trainX.shape
# NB: no need to include type_3JHN as this is redundant: this is always true when all other types are false

fitDist = model.fit(trainX, 

                    trainSet['scalar_coupling_constant'])
# Display factors to learn what is important for the prediction

fitDist.coef_
# See https://www.kaggle.com/uberkinder/efficient-metric

def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()
group_mean_log_mae(trainSet['scalar_coupling_constant'], 

                   model.predict(trainX), trainSet['type'])
# Control: this should perform better than outputing the same overfitted value for all interactions

print(group_mean_log_mae(trainSet['scalar_coupling_constant'], trainSet['scalar_coupling_constant'].median(), trainSet['type']))

print(group_mean_log_mae(trainSet['scalar_coupling_constant'], 0.85, trainSet['type']))
testX = polyFeat.transform(np.array(testSet[regressors]))[:,usefulFeatures]

resultSet = pd.DataFrame( { "id" : testSet['id'],

                            "scalar_coupling_constant" : model.predict(testX)} )
resultSet.to_csv("results.csv", index = False, header = True)
# Check content of the output file

with open("results.csv", "r") as f:

    for i, line in enumerate(f):

        print(line)

        if i > 5:

            break
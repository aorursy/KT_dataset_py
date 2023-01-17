import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

structures = pd.read_csv('../input/structures.csv')

scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')



structures_0 = structures.copy()

structures_1 = structures.copy()

structures_0.columns = structures.columns+(['']+['_0']*(len(structures.columns)-1))

structures_1.columns = structures.columns+(['']+['_1']*(len(structures.columns)-1))



train_2 = pd.merge(pd.merge(train,structures_0),structures_1)

test_2 = pd.merge(pd.merge(test,structures_0),structures_1)

train_2['distance'] = np.sqrt((train_2.x_0-train_2.x_1)**2 + (train_2.y_0-train_2.y_1)**2 + (train_2.z_0-train_2.z_1)**2)

test_2['distance'] = np.sqrt((test_2.x_0-test_2.x_1)**2 + (test_2.y_0-test_2.y_1)**2 + (test_2.z_0-test_2.z_1)**2)

reg = LinearRegression().fit(X = pd.concat([pd.get_dummies(train_2.type), train_2.distance], axis = 1), y = train_2.scalar_coupling_constant)



test_2['scalar_coupling_constant'] = reg.predict(X = pd.concat([pd.get_dummies(test_2.type), test_2.distance], axis = 1))

test_2[['id','scalar_coupling_constant']].to_csv('distance_based.csv', index = False)
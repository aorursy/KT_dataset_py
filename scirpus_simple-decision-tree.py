import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeRegressor,export_graphviz

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scipy.sparse import hstack

from sklearn.metrics import r2_score

from sklearn.model_selection import KFold

#import pydotplus as pydot

#from IPython.display import Image

#from sklearn.externals.six import StringIO

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

trainids = train.ID.ravel()

testids = test.ID.ravel()

test['y'] = -1
alldata = pd.concat([train[train.columns],test[train.columns]])

alldata.sort_values(by='X5',inplace=True)

alldata = alldata.reset_index(drop=True)
for c in alldata.columns[2:10]:

    if train[c].dtype == 'object':

        lbl = LabelEncoder()

        alldata[c] = lbl.fit_transform(list(alldata[c].values))
X0features = None

allfeatures = list(alldata.columns)

a = alldata.values

for c in ['X0']:

    ohe = OneHotEncoder()

    x = ohe.fit_transform(alldata[c].values.reshape(1,-1).T)

    X0features =  list([c+'-'+str(ci) for ci in range(x.shape[1])])

    a = hstack([a,x])

alldatasp = pd.DataFrame(a.todense())

alldatasp.columns = allfeatures+X0features

alldatasp.drop(['X0'],inplace=True,axis=1)
alldatasp = alldatasp[list(['ID']) +

                      X0features +

                      list(['X5']) +

                      list(['X118',

                           'X127',

                           'X47',

                           'X315',

                           'X311',

                           'X179',

                           'X314',

                           'X261','y'])]

alldatasp.insert(1,'SumOf',alldatasp[['X118',

                                 'X127',

                                 'X47',

                                 'X315',

                                 'X311',

                                 'X179',

                                 'X314',

                                 'X261'

                                ]].sum(axis=1))
train = alldatasp[alldatasp.ID.isin(trainids)].copy()

train.sort_values(by='ID',inplace=True)

train = train.reset_index(drop=True)

test = alldatasp[alldatasp.ID.isin(testids)].copy()

test.sort_values(by='ID',inplace=True)

test = test.reset_index(drop=True)
score = 0

splits = 10

kf = KFold(n_splits=splits)

y = train.y.ravel()

for train_index, test_index in kf.split(range(train.shape[0])):

    blind = train.loc[test_index,train.columns[1:-1]]

    vis = train.loc[train_index,train.columns[1:-1]]

    regressor = DecisionTreeRegressor(random_state=0,max_depth=4)

    regressor.fit(vis,train.y.values[train_index])

    score +=(r2_score(train.y[test_index],(regressor.predict(blind))))

print(score/splits)
regressor = DecisionTreeRegressor(random_state=0,max_depth=4)

regressor.fit(train[train.columns[1:-1]],train.y)

r2_score(train.y,regressor.predict(train[train.columns[1:-1]]))
plt.figure(figsize=(12,12))

plt.scatter(regressor.predict(train[train.columns[1:-1]]),train.y)
#dot_data = StringIO()

#export_graphviz(regressor, out_file=dot_data,feature_names=train.columns[1:-1])

#graph = pydot.graph_from_dot_data(dot_data.getvalue())

#graph.write_png('dtgraph.png')

#Image(graph.create_png())
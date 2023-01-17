import pandas as pd

buildings = pd.read_csv("../input/MN.csv")

pd.set_option('max_columns', None)

buildings.head(3)
X = buildings.loc[buildings['BldgClass'].astype(str).map(lambda v: v[0]) == 'D',

                  ['LotArea', 'BldgArea', 'ComArea', 'ResArea', 'OfficeArea', 'RetailArea',

                   'GarageArea', 'StrgeArea', 'FactryArea', 'OtherArea', 'NumFloors',

                   'UnitsTotal', 'AssessLand', 'AssessTot', 'ExemptLand', 'ExemptTot']]

y = X['UnitsTotal']

X = X.drop('UnitsTotal', axis='columns')
from sklearn.linear_model import ElasticNet

clf = ElasticNet(alpha=0.1, l1_ratio=0.5)

clf.fit(X, y)

y_pred = clf.predict(X)
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

pd.Series(y - np.abs(y_pred)).plot.hist(bins=500)

plt.gca().set_xlim([-50, 50])

pass
from sklearn.linear_model import ElasticNetCV

clf = ElasticNetCV(l1_ratio=0.5)

clf.fit(X, y)

y_pred = clf.predict(X)
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

pd.Series(y - np.abs(y_pred)).plot.hist(bins=500)

plt.gca().set_xlim([-50, 50])

pass
from sklearn.metrics import r2_score

r2_score(y, y_pred)
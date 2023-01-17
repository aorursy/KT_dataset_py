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
avocados = pd.read_csv('../input/avocado.csv')
avocados.head()
del avocados['4046']
del avocados['4225']
del avocados['4770']
avocados.groupby('type').aggregate({'Total Volume':[np.mean]})
#transform in binary values
avocados['type'] = pd.Series( np.where( avocados.type == 'conventional' , 1 , 0 ) , name = 'type' )
cities = pd.get_dummies( avocados.region , prefix='City' )
avocados = pd.concat([avocados, cities], axis=1)
avocados
y = {
    'average price': avocados['AveragePrice']
}
del avocados['AveragePrice']
del avocados['Date']
del avocados['region']
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(avocados,pd.DataFrame(y))
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(x_train, y_train)
outputs = knn.predict(x_test)
mean_squared_error(y_test, outputs)
knn.score(x_train, y_train)

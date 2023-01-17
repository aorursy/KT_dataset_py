import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

nRowsRead = 500 # prendo le prime 501 righe

# winequality-red.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

data = pd.read_csv('/kaggle/input/winequality-red.csv', delimiter=',', nrows = nRowsRead)

data.dataframeName = 'winequality-red.csv'

nRow, nCol = data.shape

print(f'There are {nRow} rows and {nCol} columns')
data.head(10)
print(data.quality.unique())
ranges = (2, 5, 6, 8) # da 3 a 5 bad, 6 average e da 7 a 8 good

classes_names = ['bad', 'average', 'good']

data['quality'] = pd.cut(data['quality'], bins = ranges, labels = classes_names)



print(data.quality.unique())
data.head(10)
from sklearn.model_selection import train_test_split



train, test = train_test_split(data, test_size = 0.3) 

from sklearn import tree



depths = [1, 3, 5, 8, 10, None]

minsamples = [1, 5, 10, 20, 40, None]



for d in depths:

    for m in minsamples:

        t = tree.DecisionTreeClassifier(max_depth = d, min_samples_leaf = m)

#        t.fit(train.drop('quality', axis = 1), train['quality']) 
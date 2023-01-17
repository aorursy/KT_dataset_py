# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/forest-cover-type-kernels-only/train.csv.zip")

test = pd.read_csv("/kaggle/input/forest-cover-type-kernels-only/test.csv.zip")
train.isnull().sum()
train_for_graphs = train

train_for_graphs['Hillshade_9am'] = train['Hillshade_9am'] - (train['Hillshade_9am'] % 10)
train_for_graphs['Elevation'] = train['Elevation'] - (train['Elevation'] % 10)
train_for_graphs['Aspect'] = train['Aspect'] - (train['Aspect'] % 10)
train_for_graphs['Slope'] = train['Slope'] - (train['Slope'] % 10)
train_for_graphs['Horizontal_Distance_To_Hydrology'] = train['Horizontal_Distance_To_Hydrology'] - (train['Horizontal_Distance_To_Hydrology'] % 10)
train_for_graphs['Vertical_Distance_To_Hydrology'] = train['Vertical_Distance_To_Hydrology'] - (train['Vertical_Distance_To_Hydrology'] % 10)
train_for_graphs['Horizontal_Distance_To_Roadways'] = train['Horizontal_Distance_To_Roadways'] - (train['Horizontal_Distance_To_Roadways'] % 10)
train_for_graphs['Hillshade_9am'] = train['Hillshade_9am'] - (train['Hillshade_9am'] % 10)
train_for_graphs['Hillshade_Noon'] = train['Hillshade_Noon'] - (train['Hillshade_Noon'] % 10)
train_for_graphs['Hillshade_3pm'] = train['Hillshade_3pm'] - (train['Hillshade_3pm'] % 10)
train_for_graphs['Horizontal_Distance_To_Fire_Points'] = train['Horizontal_Distance_To_Fire_Points'] - (train['Horizontal_Distance_To_Fire_Points'] % 10)
#t-SNE

from sklearn import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = TSNE(learning_rate=100)

train_for_graphsSmol = train_for_graphs[0:1000]
array = train_for_graphsSmol.values
X = array[:,1:]
Y = array[:,0]

transformed = model.fit_transform(X)

x_axis = transformed[:, 0]
y_axis = transformed[:, 1]

plt.scatter(x_axis, y_axis, c=Y)
plt.show()
cforest = train_for_graphs[[x for x in train_for_graphs.columns if 'Hillshade' in x] + ['Horizontal_Distance_To_Fire_Points']]
cforest[cforest.Horizontal_Distance_To_Fire_Points > 6000].groupby('Horizontal_Distance_To_Fire_Points').sum().plot()
colls = ['Horizontal_Distance_To_Fire_Points', 'Elevation', 'Slope', 'Cover_Type']
smol_forest = train[colls]

smol_forest.head()
smol_forest.groupby('Cover_Type').sum().plot(kind='bar')
X = np.array(train.drop(['Cover_Type', 'Id'], axis=1))
Y = np.array(train['Cover_Type'])
#DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)
clf.fit(X, Y)
testY = clf.predict(test.drop(['Id'], axis=1))

result = pd.DataFrame({
    'Id': test['Id'] ,
    'Cover_Type': testY
})
result
result.to_csv('submission.csv', index=False)
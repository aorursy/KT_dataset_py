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
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

df
y = df.species
df.columns
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

X = df[features]

X
X = df.drop(columns=['species'])

X
df
from sklearn.model_selection import train_test_split 

train_X, test_X, train_y, test_y = train_test_split(X, y)
train_X
train_y
test_X
test_X.shape
from sklearn.neighbors import NearestCentroid

model = NearestCentroid()
model.fit(train_X, train_y)
preds = model.predict(test_X)
preds[:5]
test_y[:5]
from sklearn.metrics import accuracy_score 

print(accuracy_score(y_true=test_y, y_pred=preds))
df.columns
test_example = pd.DataFrame(data={

    'sepal_length': [4.9],

    'sepal_width': [1.4], 

    'petal_length': [5.3], 

    'petal_width': [1.1]

})
model.predict(test_example)
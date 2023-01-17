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
train = pd.read_csv("/kaggle/input/ghouls-goblins-and-ghosts-boo/train.csv.zip")

test = pd.read_csv("/kaggle/input/ghouls-goblins-and-ghosts-boo/test.csv.zip")
train
test
train['type'].unique()
color_mapping = {'clear' : 0, 'green' : 1, 'black' : 2, 'white' : 3, 'blue' : 4, 'blood' : 5}

type_mapping = {'Ghoul' : 0, 'Goblin' : 1, 'Ghost' : 2}
train['color'] = train['color'].map(color_mapping)
train['type'] = train['type'].map(type_mapping)

test['color'] = test['color'].map(color_mapping)
train
test
x_train = train.drop('type', axis=1).values

y_train = train['type'].values



x_test = test
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', min_samples_split=4)

tree.fit(x_train, y_train)
type = tree.predict(x_test)

type
PREDICTIONS = pd.DataFrame({'id' : test['id'], 'type' : type})
PREDICTIONS
output = pd.DataFrame({'id' : test['id'], 'type' : type})

output.to_csv('submission.csv', index=False)
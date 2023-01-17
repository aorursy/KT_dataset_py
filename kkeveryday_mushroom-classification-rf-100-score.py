# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt # data visualization



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
data.head(10)
data.isnull().sum()
plt.figure(figsize=(8, 6))

sns.set_style('darkgrid')

lst_columns = list(data.columns)

for column in lst_columns:

    sns.countplot(x=column, data=data)

    plt.show()
data_t = data.drop(['veil-type'], axis=1)
data_t.head()
lst_columns = list(data_t.columns)

lst_unique = []

for column in lst_columns:

    lst_unique.append(list(data_t[column].unique()))

lst_unique
data_t['class'] = data_t['class'].map({'p': 0, 'e': 1})

data_t['cap-shape'] = data_t['cap-shape'].map({'x': 0, 'b': 1, 's': 2, 'f': 3, 'k': 4, 'c': 5})

data_t['cap-surface'] = data_t['cap-surface'].map({'s': 0, 'y': 1, 'f': 2, 'g': 3})

data_t['cap-color'] = data_t['cap-color'].map({'n': 0, 'y': 1, 'w': 2, 'g': 3, 'e': 4, 'p': 5, 'b': 6, 'u': 7, 'c': 8, 'r': 9})

data_t['bruises'] = data_t['bruises'].map({'t': 0, 'f': 1})

data_t['odor'] = data_t['odor'].map({'p': 0, 'a': 1, 'l': 2, 'n': 3, 'f': 4, 'c': 5, 'y': 6, 's': 7, 'm': 8})

data_t['gill-attachment'] = data_t['gill-attachment'].map({'p': 0, 'a': 1, 'l': 2, 'n': 3, 'f': 4, 'c': 5, 'y': 6, 's': 7, 'm': 8})

data_t['gill-spacing'] = data_t['gill-spacing'].map({'c': 0, 'w': 1})

data_t['gill-size'] = data_t['gill-size'].map({'n': 0, 'b': 1})

data_t['gill-color'] = data_t['gill-color'].map({'k': 0, 'n': 1, 'g': 2, 'p': 3, 'w': 4, 'h': 5, 'u': 6, 'e': 7, 'b': 8, 'r': 9, 'y': 10, 'o': 11})

data_t['stalk-shape'] = data_t['stalk-shape'].map({'e': 0, 't': 1})

data_t['stalk-root'] = data_t['stalk-root'].map({'e': 0, 'c': 1, 'b': 2, 'r': 3, '?': 4})

data_t['stalk-surface-above-ring'] = data_t['stalk-surface-above-ring'].map({'s': 0, 'f': 1, 'k': 2, 'y': 3})

data_t['stalk-surface-below-ring'] = data_t['stalk-surface-below-ring'].map({'s': 0, 'f': 1, 'y': 2, 'k': 3})

data_t['stalk-color-above-ring'] = data_t['stalk-color-above-ring'].map({'w': 0, 'g': 1, 'p': 2, 'n': 3, 'b': 4, 'e': 5, 'o': 6, 'c': 7, 'y': 8})

data_t['stalk-color-below-ring'] = data_t['stalk-color-below-ring'].map({'w': 0, 'p': 1, 'g': 2, 'b': 3, 'n': 4, 'e': 5, 'y': 6, 'o': 7, 'c': 8})

data_t['veil-color'] = data_t['veil-color'].map({'w': 0, 'n': 1, 'o': 2, 'y': 3})

data_t['ring-number'] = data_t['ring-number'].map({'o': 0, 't': 1, 'n': 2})

data_t['ring-type'] = data_t['ring-type'].map({'p': 0, 'e': 1, 'l': 2, 'f': 3, 'n': 4})

data_t['spore-print-color'] = data_t['spore-print-color'].map({'k': 0, 'n': 1, 'u': 2, 'h': 3, 'w': 4, 'r': 5, 'o': 6, 'y': 7, 'b': 8})

data_t['population'] = data_t['population'].map({'s': 0, 'n': 1, 'a': 2, 'v': 3, 'y': 4, 'c': 5})

data_t['habitat'] = data_t['habitat'].map({'u': 0, 'g': 1, 'm': 2, 'd': 3, 'p': 4, 'w': 5, 'l': 6})
data_t.tail(7)
X = data_t.drop(['class'], axis=1)

y = data_t['class']
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)
rfc_train_preds = rfc.predict(X_train)

train_score = round(accuracy_score(y_train, rfc_train_preds) * 100, 2)

print('Accuracy on training dataset =', train_score)
rfc_test_preds = rfc.predict(X_test)

test_score = round(accuracy_score(y_test, rfc_test_preds) * 100, 2)

print('Accuracy on test dataset =', test_score)
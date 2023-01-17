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
df = pd.read_csv(os.path.join(dirname, filename))
df = df.drop('Id',axis=1)
df.head(2)
df.Species.unique()
def species_convertor(x):
    if x == 'Iris-setosa':
        return 1
    elif x == 'Iris-versicolor':
        return 2
    else:
        return 3
df['numerical_targer'] = df.Species.apply(species_convertor)
df = df.drop('Species',axis=1)
df.head(1)
X_values = df.iloc[:,0:4].values
Y_values = df.iloc[:,4:5].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_values, Y_values, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier
clr = DecisionTreeClassifier(max_depth=6)
clr.fit(X_train, y_train)
predictions = clr.predict(X_test)
print('Accuracy is {}%'.format(int(sum(y_test == predictions.reshape(-1,1))/len(y_test)*100)))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_data.describe()
y=train_data['label']
X=train_data.copy()
del X['label']
print(X)
np.shape(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
from sklearn.preprocessing import normalize
X_train_norm=normalize(X_train)
X_test_norm=normalize(X_test)
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='adam',hidden_layer_sizes=350,alpha=1e-04)
clf.fit(X_train_norm,y_train)
from sklearn.metrics import accuracy_score

preds=clf.predict(X_test_norm)
print(accuracy_score(y_test,preds))
X_pred=test_data
from sklearn.preprocessing import normalize

result = clf.predict(normalize(X_pred))

df=pd.DataFrame(result)
df.index+=1
print(result.shape)
filename = 'DigitalPredictions2.csv'
df.to_csv(filename,index=True,header=["Label"],index_label=["ImageId"])
print('Saved file: ' + filename)

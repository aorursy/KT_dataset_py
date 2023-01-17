# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # For graphical representation 
import seaborn as sns 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
a = pd.read_csv('../input//iris/Iris.csv')
iris = pd.DataFrame(a)
iris.head()
iris.drop('Id',axis=1,inplace=True)
iris.plot(kind = 'box', subplots = True, layout = (2, 2), sharex = False, sharey = False)
plt.show()
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
a = pd.read_csv('../input//iris/Iris.csv', header = None)
i = pd.DataFrame(a)
iris = i.values
X = iris[1:, 1:5].astype(float)
Y = iris[1:, 5]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# One Hot Encode
#encoded_Y
y_dummy = np_utils.to_categorical(encoded_Y)
y_dummy
def deepml_model():
    # Model Creation
    deepml = Sequential()
    deepml.add(Dense(8, input_dim=4, activation='relu'))
    deepml.add(Dense(3, activation='softmax'))
    # Model Compilation
    deepml.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return deepml
estimate = KerasClassifier(build_fn=deepml_model, epochs=200, batch_size=5, verbose=0)
estimate
seed=7
k_fold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimate, X, y_dummy, cv=k_fold)
print("Model: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
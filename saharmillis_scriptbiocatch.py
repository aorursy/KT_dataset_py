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
import warnings
warnings.filterwarnings('ignore')

import pickle

with open("../input/exercise_to_sahar.p", "rb") as f:
    dump = pickle.load(f)
    X_train_original,Y_train_original,X_test_original = dump[0], dump[1], dump[2]
    

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def get_report(a,b):
    target_names = ['class 0', 'class 1']
    return classification_report(a, b, target_names=target_names)

print(get_report(Y_train_original,Y_train_original))    

from imblearn.pipeline import Pipeline 
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split as TTS

pca = PCA()
rfc = RFC(n_jobs=-1,warm_start=True)
smt = SMOTETomek(ratio='minority',n_jobs=-1)
                           
pipeline = Pipeline([('smt', smt), ('pca', pca), ('rfc', rfc)])

X_train, X_test, y_train, y_test = TTS(X_train_original, Y_train_original,train_size=0.8,random_state=42)

pipeline.fit(X_train, y_train) 

y_hat = pipeline.predict(X_test)
print(classification_report(y_test, y_hat))
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
df=pd.read_csv("/kaggle/input/MLChallenge-2/final.csv")
df_test=pd.read_csv("/kaggle/input/MLChallenge-2/Test.csv")

df_test.shape
df.head(10)
df.shape

df_new=df.drop("target",axis=1)
df_new.head()
#x=df.drop("target", axis=1)
x_train=df_new[["oldpeak","age"]]
y_train=df_new[["thal"]]
x_test=df_test[["oldpeak","age"]]
y_test=df_test[["thal"]]
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,10))
clf.fit(x_train,y_train)
y_pred=clf.predict(x_train)
from sklearn.metrics import classification_report 
classification_report(y_pred, y_train)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred, y_train)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred, y_train)
clf.fit(x_test, y_test)
y_pred_test=clf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred_test, y_test)



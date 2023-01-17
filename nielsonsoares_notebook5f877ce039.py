# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/train.csv')

df2 = pd.read_csv('../input/test.csv')

parent_data = df1.copy()  

print (df1.shape)
ID = df1.pop('id')

y = df1.pop('species')

X = df1
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_encoded = le.fit_transform(y)
from sklearn.preprocessing import scale

X_scaled = scale(X)
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score



clf = MLPClassifier()



score = cross_val_score(clf, X_scaled, y_encoded)
score
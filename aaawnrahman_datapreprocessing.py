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
data = pd.read_csv("../input/thesisda/final.csv")

print(data)


X= np.array(data.iloc[:,1:25].values)

Y=np.array(data.iloc[:,25].values)

print(Y)
val = [5,6,7,8,18,19,20,21,22,23]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

for v in val :

    X[:, v] = labelencoder_X.fit_transform(X[:, v])



labelencoder_y = LabelEncoder()

Y = labelencoder_y.fit_transform( Y)



print(X)

print(Y)

from sklearn.preprocessing import Imputer



miss= [1,2,3,4,9,10,11,12,13,14,15,16,17]

print(X[:2])

calcul=X[:,v ].reshape(-1,1)



imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 1:5])

z=np.array(X[:, 1:3])

X[:, 1:5] = imputer.transform(X[:, 1:5])

z1=np.array(X[:, 1:5])



imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X[:, 9:18])

z=np.array(X[:, 9:18])

X[:, 9:18] = imputer.transform(X[:, 9:18])

z1=np.array(X[:, 9:18])

print(X[:2])
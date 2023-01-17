import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/pima_data.csv")
array = data.values
X = array[:,0:8]
Y = array[:,8]
print(X)
#scale Features from 0 to 1
transforms = MinMaxScaler(feature_range=(0,1))
scaler_X =  transforms.fit_transform(X)
print(scaler_X)
#StandardScaler
transforms = StandardScaler()
standardScale_X = transforms.fit_transform(X)
print(standardScale_X)
transforms = Normalizer()
normalizer_x = transforms.fit_transform(X)
print(normalizer_x)
transforms = Binarizer(threshold=0.5)
binarizer_x = transforms.fit_transform(X)
print(binarizer_x)

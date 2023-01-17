import numpy as np
import pandas as pd
import os
df = pd.read_csv("../input/train.csv")
df.count()
df = df.drop('Cabin', axis = 1)
df.describe()
from sklearn.preprocessing import Imputer

#Split the data and get values for sklearn
x = df.iloc[:,2:].values
y = df.iloc[:,1].values

imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)

x[:,3] = imp.fit_transform(np.asmatrix(x[:,3]))
x
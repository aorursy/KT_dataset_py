import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error
train = pd.read_csv('../input/train.csv')
train.head()
A = train[train.columns[2:]].values
b = train.target.values
Q,R = np.linalg.qr(A.T)
RTinv = np.linalg.pinv(R.T)
U = np.dot(RTinv,b)
predictions = np.dot(R.T,U).clip(train.target.min(),train.target.max()) #We have no conditions so just keep the values in the range of min and max for target
np.sqrt(mean_squared_error(np.log1p(b),np.log1p(predictions)))
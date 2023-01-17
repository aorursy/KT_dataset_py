import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train = pd.read_csv('/kaggle/input/integer-sequence-learning/train.csv.zip')

test = pd.read_csv('/kaggle/input/integer-sequence-learning/test.csv.zip')
train.head()
val = train.Sequence.values[0].split(',')
val
val_shaped = np.reshape(np.array(val),(7, 2))

val_shaped

X = [int(i[0]) for i in val_shaped]

y = [int(i[1]) for i in val_shaped]

X_shaped = np.reshape(np.array(X),(7,1))

y_shaped = np.reshape(np.array(y),(7,1))
X_shaped
y_shaped
from sklearn.preprocessing import PolynomialFeatures 

from sklearn.linear_model import LinearRegression

  

poly = PolynomialFeatures(len(X))

X_poly = poly.fit_transform(X_shaped)

  

# fit the transformed features to Linear Regression

poly_model = LinearRegression()

poly_model.fit(X_poly, y_shaped)

predicted = poly_model.predict(X_poly)

predicted
y_shaped
predicted
y_shaped[6][0] -predicted[6][0]


    
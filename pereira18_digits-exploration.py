import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split 

from matplotlib import pyplot as plt



from sklearn.datasets import load_digits
digits = load_digits()
dir(digits)
df = pd.DataFrame(digits.data, digits.target)

df.head()
df['target'] = digits.target

df.head()
X = df.drop(['target'],axis='columns')

y = df.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from sklearn.svm import SVC

rbf_model = SVC(C=20, kernel='rbf')
len(X_train)
len(X_test)
rbf_model.fit(X_train, y_train)

rbf_model.score(X_test, y_test)
linear_model = SVC(kernel='linear')

linear_model.fit(X_train,y_train)
linear_model.score(X_test,y_test)
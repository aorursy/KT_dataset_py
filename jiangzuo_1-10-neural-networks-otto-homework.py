import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt


X = data[columns]
y = np.ravel(data['target'])




num_fea = X.shape[1]


model.intercepts_
print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)
pred = model.predict(X)
pred

sum(pred == y) / len(y)


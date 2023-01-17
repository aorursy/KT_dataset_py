import numpy as np
import pandas as pd
from matplotlib import pyplot 
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as Metrics
import warnings as w
import matplotlib

w.filterwarnings('ignore')
%matplotlib inline
X, y = sklearn.datasets.load_digits(return_X_y=True)
fig, ax_array = pyplot.subplots(5, 5, figsize=(5, 5))
fig.subplots_adjust(wspace=0.025, hspace=0.025)

ax_array = ax_array.ravel()

for i, ax in enumerate(ax_array):
    
    h = ax.imshow(X[i].reshape(8, 8), cmap='Greys')
    ax.axis('off')
model = LogisticRegression()
model.fit(X, y)
print("Accuracy: {}%".format(np.multiply(model.score(X, y), 100)))
Predictions = model.predict(X)
index = 78

img = X[index].reshape(8, 8)
title = "You predicted 'y={}'".format(Predictions[index])
pyplot.imshow(img, cmap='Greys')
pyplot.title(title)
pass
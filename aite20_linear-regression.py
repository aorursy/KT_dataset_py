import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets as Datasets
import sklearn.metrics as Metrics
import warnings as w

w.filterwarnings('ignore')
X, y = Datasets.load_boston(return_X_y=True)
Dataset = Datasets.load_boston()
feature_names = Dataset.feature_names
ax = plt.axes()
feature = 2
ax.scatter(X[:, feature], y, label='Dataset', edgecolor='darkred', facecolor='blue')
ax.set_xlabel(feature_names[feature])
ax.set_ylabel('Price')
ax.set_title('Boston dataset')
ax.legend(loc='upper right', fancybox=True, shadow=True, frameon=True, framealpha=0.9)
pass
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
print("Training accuracy is: {:.2f}%".format(np.multiply(model.score(X, y), 100)))
y_pred = model.predict(X)
y_pred = np.linspace(np.min(y_pred), np.max(y_pred), X.shape[0])
feature = 0
x = np.linspace(np.min(X[:, feature]), np.max(X[:, feature]), X.shape[0])

ax = plt.axes()
ax.scatter(X[:, feature], y, label='Dataset', edgecolor='darkred', facecolor='blue')
ax.plot(x, y_pred, label='Bestfit line', linewidth=1.2, color='red')
ax.set_xlabel(feature_names[feature])
ax.set_ylabel('Price')
ax.legend(loc='upper right', fancybox=True, shadow=True, frameon=True, framealpha=0.9)
pass
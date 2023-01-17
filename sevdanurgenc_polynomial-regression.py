import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression

m = 100
# 100 tane dataset büyüklüğü vardır.

X = 6 * np.random.rand(m, 1) - 3
print(X)
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
plt.figure(dpi=200)
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel('y', rotation=0, fontsize=18)
plt.scatter(X, y)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#önce pipeline ile linear regression haline getirilmesi gerekir.
#linear için linespace kullanıyoruz.
#include bias false ile bias eklemiyoruz.
#degree ile x karenin ikisini belirtiyoruz. ikinci derece oluşu.

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

polynomial_regression.fit(X, y)
X_new=np.linspace(-3, 3, 100).reshape(100, 1)
y_newbig = polynomial_regression.predict(X_new)
plt.figure(dpi=200)
plt.scatter(X, y)
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.plot(X_new, y_newbig, "r-", linewidth=2, label="Predictions")
plt.legend(loc="upper left", fontsize=14)
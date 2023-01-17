from sklearn.datasets import load_boston

boston = load_boston()
print (boston.data.shape)
type (boston)
from sklearn.datasets import load_boston

boston = load_boston()
import pandas as pd

bos = pd.DataFrame(boston.data)

print (bos.head())
bos['PRICE'] = boston.target

X = bos.drop('PRICE', axis=1)

Y = bos['PRICE']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state=5)

print (X_train.shape)

print (X_test.shape)

print (y_train.shape)

print (y_test.shape)
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt



lm = LinearRegression()

lm.fit(X_train, y_train)



Y_pred = lm.predict(X_test)

plt.scatter(y_test, Y_pred)

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Pridicted prices: $\hat\{Y}_i$")

plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

plt.show()
delta_y = y_test - Y_pred



import seaborn as sns

import numpy as np

sns.set_style('whitegrid')

sns.kdeplot(np.array(delta_y),bw=0.5)

plt.show()
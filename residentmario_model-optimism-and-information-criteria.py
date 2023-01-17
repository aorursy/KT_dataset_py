import pandas as pd

import numpy as np

df = pd.read_csv("../input/jcpenney_com-ecommerce_sample.csv")

df = (df[['list_price', 'sale_price']]

        .applymap(lambda v: str(v)[:4]).dropna().astype(np.float64)).dropna()

df.head()
X = df.iloc[:, 0].values[:, np.newaxis]

y = df.iloc[:, 1].values
from sklearn.linear_model import LassoLarsIC



clf = LassoLarsIC()

clf.fit(X, y)

y_hat = clf.predict(X)
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



sns.jointplot(y, y_hat)

plt.gcf().suptitle('JC Penny Sale Price Predicted via List Price')

pass
clf.alpha_
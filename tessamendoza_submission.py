import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train_data.head()
train_data.columns
train_data['SalePrice'].describe()
sns.distplot(train_data["SalePrice"])
#correlation matrix

corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train_data[cols].values.T)

sns.set(font_scale=1.2)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(train_data[cols], height = 2.5)

plt.show()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge



features = ['OverallQual', 'GrLivArea', 'YearBuilt']

label = 'SalePrice'



X_train = train_data[features]

y_train = train_data[label]



model = Pipeline([

                    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),

                    ("std_scaler", StandardScaler()),

                    ("regul_reg", Ridge(alpha=0.05, solver="cholesky")),

                ])

model.fit(X_train, y_train)
X_test = test_data[features]

y_pred = model.predict(X_test)
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': y_pred})

submission.to_csv('tessa_mendoza.csv', index=False)
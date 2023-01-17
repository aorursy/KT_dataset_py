import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/advertising.csv/Advertising.csv")

df.head()
# Understand my variables

print(df.columns)

df.describe()
# Dropping redundant variables

df = df.copy().drop(['Unnamed: 0'],axis=1)
# Sanity check

print(df.columns)

print(df.shape)

df.isna().sum()
# Calculate correlation matrix

corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))

# Scatterplot of all variables

sns.pairplot(df)
df.boxplot("TV")
df.boxplot("radio")
df.boxplot("newspaper")
X = df.loc[:, df.columns != 'sales']

y = df['sales']
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error as mae



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

model = RandomForestRegressor(random_state=1)

model.fit(X_train, y_train)

pred = model.predict(X_test)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
import statsmodels.formula.api as sm

modelA = sm.ols(formula="sales~TV+radio+newspaper", data=df).fit()

modelB = sm.ols(formula="sales~TV+radio+newspaper", data=df).fit()

print(modelA.summary())

print(modelB.summary())
y_pred = modelB.predict()

labels = df['sales']

df_temp = pd.DataFrame({'Actual': labels, 'Predicted':y_pred})

df_temp.head()
from matplotlib.pyplot import figure

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')



y1 = df_temp['Actual']

y2 = df_temp['Predicted']



plt.plot(y1, label = 'Actual')

plt.plot(y2, label = 'Predicted')

plt.legend()

plt.show()
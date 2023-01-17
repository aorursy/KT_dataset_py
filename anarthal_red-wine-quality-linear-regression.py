# Let's start with some imports and loading our dataset



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.preprocessing import StandardScaler, PolynomialFeatures



df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df['quality'].describe()
sns.countplot(x='quality', data=df)
corr = df.corr()

idx = corr['quality'].abs().sort_values(ascending=False).index[:5]

idx_features = idx.drop('quality')

sns.heatmap(corr.loc[idx, idx])
_, ax = plt.subplots(2, 2, figsize=(20, 10))

for var, axis in zip(idx_features, ax.flatten()):

    df[var].plot.hist(ax=axis)

    axis.set_xlabel(var)
df.describe()
sns.pairplot(df, vars=idx)
_, ax = plt.subplots(2, 2, figsize=(20, 10))

for i, var in enumerate(idx.drop('quality')):

    sns.boxplot(x='quality', y=var, data=df, ax=ax.flatten()[i])
def plot_learning_curves(X, y, model):

    train_sizes, train_scores, cv_scores = learning_curve(model, X, y)

    train_scores = np.mean(train_scores[1:], axis=1)

    cv_scores = np.mean(cv_scores[1:], axis=1)

    plt.figure(figsize=(10,10))

    plt.plot(train_sizes[1:], train_scores, label='Train')

    plt.plot(train_sizes[1:], cv_scores, label='CV')

    plt.xlabel('Sample size')

    plt.ylabel('R2')

    plt.legend()

    

def train(X, y, model, poly_degree=None):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    if poly_degree is not None:

        pol = PolynomialFeatures(poly_degree, include_bias=False)

        X_train = pol.fit_transform(X_train)

        X_test = pol.transform(X_test)

    model.fit(X_train, y_train)

    r2_train = model.score(X_train, y_train)

    r2_test = model.score(X_test, y_test)

    print('r2_train = {:.3f}, r2_test={:.3f}'.format(r2_train, r2_test))

    plot_learning_curves(X, y, model)
# Simple linear regression

features = df.drop(columns='quality')

X = features.copy()

y = df['quality']

train(X, y, LinearRegression())
# Let's try adding some polynomic features

train(X, y, LinearRegression(), poly_degree=2)
train(X, y, Ridge(alpha=2.0), poly_degree=2)
feature_subset = df[idx].drop(columns='quality')

train(feature_subset, y, Ridge(5.0), poly_degree=2)
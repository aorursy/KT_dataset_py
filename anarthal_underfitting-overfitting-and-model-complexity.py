import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.datasets import make_moons

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import xgboost as xgb



sns.set()
X, y = make_moons(2000, random_state=0, noise=0.22)

X = StandardScaler().fit_transform(X) # Useful for polynomial features

xvar, yvar, label = 'height', 'ear_length', 'survived'

df = pd.DataFrame({xvar: X[:, 0], yvar: X[:, 1], label: y})
plt.figure(figsize=(15, 8))

sns.scatterplot(data=df, x=xvar, y=yvar, hue=label, palette={0.0: 'red', 1.0: 'green'});
X_train, X_test, y_train, y_test = train_test_split(df[[xvar, yvar]], df[label], random_state=0)
plt.figure(figsize=(15, 8))

sns.scatterplot(x=X_train[xvar], y=X_train[yvar], hue=y_train, palette={0.0: (1.0, 0.7, 0.7), 1.0: (0.7, 1.0, 0.7)}, legend=False)

sns.scatterplot(x=X_test[xvar], y=X_test[yvar], hue=y_test, palette={0.0: 'red', 1.0: 'green'}, legend=False)

plt.title('Train-test split');
def plot_decision_boundary(model, df):

    xmin, xmax = -2.5, 2.5

    ymin, ymax = -2.5, 2.5

    xstep = 0.01

    ystep = 0.01

    

    xx, yy = np.meshgrid(np.arange(xmin, xmax+xstep, xstep), np.arange(ymin, ymax+ystep, ystep))

    meshdf = pd.DataFrame({xvar: xx.ravel(), yvar: yy.ravel()})

    Z = model.predict(meshdf).reshape(xx.shape)



    plt.figure(figsize=(16, 8))

    plt.pcolormesh(xx, yy, Z, cmap=ListedColormap([(1.0, 0.7, 0.7), (0.7, 1.0, 0.7)]))

    sns.scatterplot(data=df, x=xvar, y=yvar, hue=label, palette={0.0: 'red', 1.0: 'green'});
model = LogisticRegression()

model.fit(X_train, y_train)

print('Train accuracy: ', model.score(X_train, y_train))

print('Test  accuracy: ', model.score(X_test, y_test))

plot_decision_boundary(model, df)
model = Pipeline([

    ('poly', PolynomialFeatures(3, include_bias=False)),

    ('model', LogisticRegression(max_iter=100000))

])

model.fit(X_train, y_train)

print('Train accuracy: ', model.score(X_train, y_train))

print('Test  accuracy: ', model.score(X_test, y_test))

plot_decision_boundary(model, df)
model = Pipeline([

    ('poly', PolynomialFeatures(15)),

    ('model', LogisticRegression(max_iter=100000, solver='newton-cg', penalty='none'))

])

model.fit(X_train, y_train)

print('Train accuracy: ', model.score(X_train, y_train))

print('Test  accuracy: ', model.score(X_test, y_test))

plot_decision_boundary(model, df)
model = xgb.XGBClassifier()

model.fit(X_train, y_train)

print('Train accuracy: ', model.score(X_train, y_train))

print('Test  accuracy: ', model.score(X_test, y_test))

plot_decision_boundary(model, df)
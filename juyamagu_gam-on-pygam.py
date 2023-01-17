!python -m pip install pygam
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



# https://www.kaggle.com/juyamagu/glm-on-statsmodels/ と同様にデータを揃える

boston = load_boston()

df = pd.DataFrame(boston.data, columns=boston.feature_names)

df['PRICE'] = boston.target

df = df[['RM', 'PTRATIO', 'LSTAT', 'PRICE']]



X = df.drop('PRICE', axis=1)

y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2019)
from pygam import s, f, LinearGAM



gam = LinearGAM(s(0) + s(1) + s(2)).fit(X_train, y_train)

gam.summary()
fig, axes = plt.subplots(1,3, figsize=(10,5));



for i, (ax, title) in enumerate(zip(axes, X_train.columns)):

    XX = gam.generate_X_grid(term=i)

    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))

    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

    ax.set_title(title);
y_pred = gam.predict(X_test)

print('MSE:', mean_squared_error(y_test, y_pred))
pred_intervals = gam.prediction_intervals(X_test)

print(y_pred[:5])

print(pred_intervals[:5])
def plot_model(model):

    fig, axes = plt.subplots(1, 3, figsize=(10,5));

    for i, (ax, title) in enumerate(zip(axes, X_train.columns)):

        XX = model.generate_X_grid(term=i)

        ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX))

        ax.plot(XX[:, i], model.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')

        ax.set_title(title);
normal_model = LinearGAM(s(0) + s(1) + s(2)).fit(X_train, y_train)

plot_model(normal_model)



wired_model = LinearGAM(s(0, n_splines=300) + s(1, n_splines=4) + s(2)).fit(X_train, y_train)

plot_model(wired_model)
normal_model = LinearGAM(s(0) + s(1) + s(2)).fit(X_train, y_train)

plot_model(normal_model)



wired_model = LinearGAM(s(0, lam=1e-5) + s(1, lam=1e+5) + s(2)).fit(X_train, y_train)

plot_model(wired_model)
normal_model = LinearGAM(s(0) + s(1) + s(2)).fit(X_train, y_train)

plot_model(normal_model)



wired_model = LinearGAM(s(0, basis='cp') + s(1, basis='cp') + s(2, basis='cp')).fit(X_train, y_train)

plot_model(wired_model)
from pygam import GammaGAM



model = GammaGAM(s(0) + s(1) + s(2)).fit(X_train, y_train)

plot_model(model)
y_pred = model.predict(X_test)

print('MSE:', mean_squared_error(y_test, y_pred))
from pygam import GammaGAM, l



model = GammaGAM(l(0) + l(1) + l(2)).fit(X_train, y_train)

plot_model(model)



y_pred = model.predict(X_test)

print('MSE:', mean_squared_error(y_test, y_pred))
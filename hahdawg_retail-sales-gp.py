import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab

import seaborn as sns

import sklearn.gaussian_process as gp

import sklearn.gaussian_process.kernels as gpk

import sklearn.model_selection as ms

pylab.rcParams["figure.figsize"] = (12., 8.)



## Fetch retail sales data

data = pd.read_csv("../input/RSAFSNA.csv")

data["DATE"] = pd.to_datetime(data["DATE"])

print(data.head())

ds = data.DATE.values

xs = np.arange(len(ds))

ys = data.RSAFSNA.values

nobs = len(ys)



## Fit a GP

ls_lim = 12*10

short_trend = gpk.ConstantKernel()*gpk.Matern(nu=1.5, length_scale_bounds=(1e-5, ls_lim))

long_trend = gpk.ConstantKernel()*gpk.Matern(nu=1.5, length_scale_bounds=(ls_lim, 1e5))

trend = short_trend + long_trend

seas = gpk.ConstantKernel()*gpk.ExpSineSquared(periodicity=12., periodicity_bounds=(12., 12.))

noise = gpk.WhiteKernel(noise_level=0.1)

kernel =  trend + seas + noise

y = ys.copy()

X = xs.reshape(-1, 1)

X_tr, _, y_tr, _, idx_tr, _ = ms.train_test_split(X, y, np.arange(nobs), test_size=0.5)

model = gp.GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)

model.fit(X_tr, np.log(y_tr))



## Make some predictions

num_pred_years = 5

num_pred_months = 12*5

ds_pred = pd.date_range(start=ds.min(), periods=len(ds) + num_pred_months, freq="MS")

X_pred = np.arange(len(ds_pred)).reshape(-1, 1)

y_hat = np.exp(model.predict(X_pred))



print(model.kernel_)



## Plot the data

fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(ds, ys, label="y", alpha=0.5)

ax.plot(ds_pred, y_hat, label="y_hat", lw=4.0, alpha=0.5)

ax.plot(ds[idx_tr], y_tr, label="y_tr", linestyle="", marker="^", ms=5.)

plt.legend(loc=0)

plt.show()





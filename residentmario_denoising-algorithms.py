import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
train = pd.read_parquet("../input/train.parquet")
train.iloc[:, 0:3].plot(figsize=(12, 8))
plt.axis('off'); pass
df = train.iloc[:, 0:3]
df.rolling(100).mean().plot(figsize=(12, 8))
plt.axis('off'); pass
df.rolling(100, win_type='gaussian').mean(std=df.std().mean()).plot(figsize=(12, 8))
plt.axis('off'); pass
# pd.Series(np.repeat([0., 1., 0.], 10)).plot()
def apply_convolution(sig, window):
    conv = np.repeat([0., 1., 0.], window)
    filtered = signal.convolve(sig, conv, mode='same') / window
    return filtered

df.apply(lambda srs: apply_convolution(srs, 100)).plot(figsize=(12, 8))
plt.axis('off'); pass
# TODO: recipe based on https://scipy-cookbook.readthedocs.io/items/Interpolation.html
# I had difficulty getting this working because I did not know how to interpret the result
# of signal.cspline1d, which claims to generate coefficients, not values; but those 
# coefficients cannot be immediately mapped onto data values.

# coef = df.apply(lambda srs: signal.cspline1d(srs[::50].values, 10))
# df_approx = signal.cspline1d_eval(coef.iloc[:, 0].values.astype('float64'), df.index.values.astype('float64'))
# coef.head(100).apply(lambda t: t[0]*t.name**3 + t[1]*t.name**2 + t[2], axis='columns').plot()
# ^ goes to \infty
df.apply(lambda srs: signal.savgol_filter(srs.values, 99, 3)).plot(figsize=(12, 8))
plt.axis('off'); pass
from sklearn.neighbors import KNeighborsRegressor

clf = KNeighborsRegressor(n_neighbors=100, weights='uniform')
clf.fit(df.index.values[:, np.newaxis], 
        df.iloc[:, 0])
y_pred = clf.predict(df.index.values[:, np.newaxis])
ax = pd.Series(df.iloc[:, 0]).plot(color='lightgray')
pd.Series(y_pred).plot(color='black', ax=ax, figsize=(12, 8))
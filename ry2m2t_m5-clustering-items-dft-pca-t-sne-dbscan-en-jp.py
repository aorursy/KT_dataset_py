import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import rcParams; rcParams["figure.figsize"] = (16, 8)

import seaborn as sns; sns.set(context="poster")
!pip install opentsne
from openTSNE.sklearn import TSNE
%%time

path = "/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv"

df = pd.read_csv(path)
df.head()
D_MAX = 1941

D_MIN = 1
cols_dim = df.loc[:, :"state_id"].columns.tolist()

cols_d = df.loc[:, f"d_{D_MIN}":].columns.tolist()
df_unit = df.loc[:, cols_dim + cols_d]
df_unit
df_unit = df_unit.groupby("item_id")[cols_d].sum()
df_unit
from scipy.fft import fft, ifft

from scipy.signal import get_window
N = 1024

dt = 1  # day

w = get_window("flattop", N)



x_f = np.linspace(0.0, 1.0 / (2.0 * dt), N//2)

data = df_unit.iloc[:, -N:].to_numpy()
%time data_freq = fft(w * data)
%time spectrum = 2.0 / N * np.abs(data_freq[:, 0:(N // 2)])
fig, ax = plt.subplots()

ax.plot(x_f, spectrum[0], lw=2)

ax.set(xlabel="frequency (1/day)", ylabel="spectrum")
d_f = 1 / (N * dt)

d_f  # in day^(-1)
f_max = 1 / (2 * dt)

f_max  # in day^(-1)
wtypes = ["hamming", "hann", "flattop", "blackman"]
fig, ax = plt.subplots(figsize=(16, 8))

for wtype in wtypes:

    w = get_window(wtype, N)

    ax.plot(w, label=wtype)

    ax.set(xlabel="day (last 1024 days)", ylabel="weight")

    ax.legend()
fig, ax = plt.subplots(figsize=(16, 8))

for wtype in wtypes:

    x_f = np.linspace(0.0, 1.0 / (2.0 * dt), N//2)

    w = get_window(wtype, N)

    y_f = fft(w)

    s_f = 2.0 / N * np.abs(y_f[0:(N // 2)])  # cut at Nyquist freq.

    

    ax.plot(x_f, s_f, lw=1, label=wtype)

    ax.set(xlim=(0, 0.05))

    ax.set(yscale="log")

    ax.set(xlabel="frequency (1/day)", ylabel="spectrum")

    ax.legend()
x = df_unit.iloc[0, -N:].to_numpy()



fig, ax = plt.subplots(figsize=(16, 8))

for wtype in wtypes:

    w = get_window(wtype, N)

    y_f = fft(w * x)

    s_f = 2.0 / N * np.abs(y_f[0:(N // 2)])  # cut at Nyquist freq.

    

    ax.plot(x_f, s_f, lw=1, label=wtype)

    ax.set(yscale="log")

    ax.set(xlabel="frequency (1/day)", ylabel="spectrum")

    ax.set(xlim=(0, 0.05))

    ax.legend()
from sklearn.decomposition import PCA
pca = PCA(

    n_components=None,

    random_state=0,

)
spectrum_pca = pca.fit_transform(spectrum)
pca.components_[0][0:100]
from matplotlib import ticker
_data = pca.explained_variance_ratio_.cumsum()



fig, ax = plt.subplots()

ax.plot(_data, marker="o")

ax.set(xlim=(-.5, 20))

ax.set(xlabel="# of dimensions", ylabel="explained variance ratio")

ax.xaxis.set(major_locator=ticker.FixedLocator(np.arange(0, 20, 1)))
N_PCA_COMPONENTS = 5
spectrum_pca = spectrum_pca[:, :N_PCA_COMPONENTS]
spectrum_pca
from openTSNE.sklearn import TSNE
from tqdm import tqdm
dofs = [0.5, 0.7, 0.9, 1]

perps = [10, 20, 30, 40, 50]

params = dict(

    n_components=2,

    negative_gradient_method="bh",

    initialization="pca",

    random_state=0,

    n_jobs=-1,

)
%%time



_data = spectrum_pca

estimators = []



n_perps = len(perps)

n_dofs = len(dofs)

fig, axes = plt.subplots(nrows=n_perps, ncols=n_dofs, figsize=(8 * n_dofs, 8 * n_perps))



for i, perp in enumerate(tqdm(perps)):

    for j, dof in enumerate(tqdm(dofs)):

    

        ax = axes[i, j]

        ax.set(title=f"perplexity={perp}, dof={dof}")

        ax.set(xlabel="axis 0", ylabel="axis 1")



        estimator = TSNE(dof=dof, perplexity=perp, **params)

        transformed = estimator.fit_transform(_data)

        estimators.append(estimator)



        ax.scatter(*(transformed.T), s=3)



fig.tight_layout()
estimator = estimators[9]

transformed = estimator.transform(_data)



fig, ax = plt.subplots(figsize=(12, 12))

ax.scatter(*(transformed.T), s=3)

ax.set(xlabel="axis 0", ylabel="axis 1")
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN
transformed = StandardScaler().fit_transform(transformed)

clustered = DBSCAN(eps=0.2).fit_predict(transformed)



fig, ax = plt.subplots(figsize=(16, 16))



for i in np.unique(clustered):

    _data = transformed[clustered == i].T

    ax.scatter(*_data, s=10, label=f"cluster {i}")



ax.set(xlabel="axis 0", ylabel="axis 1")

ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1));
clusters = np.unique(clustered)[1:]  # omit '-1'

annots = [

    (1/365.25, "yearly"),

    (1/30.5, "monthly"),

    (1/7, "weekly"),

    (1/3.5, "semiweekly"),

]



fig, axes = plt.subplots(len(clusters), 1, figsize=(24, 6 * len(clusters)))

ylim = (0, 20)



for i, cluster in enumerate(tqdm(clusters)):

    ax = axes[i]

    _data = spectrum[clustered == cluster].T

    n_series = (clustered == cluster).sum()

    

    ax.plot(x_f, _data, color="navy", alpha=0.2, lw=1)

    

    for pos, label in annots:

        ax.axvline(pos, color="red", alpha=0.25)  # constant term

        ax.annotate(label, xy=(pos, ylim[1]), textcoords="offset points", xytext=(2, -20), fontsize="small")



    ax.set(ylim=ylim)

    ax.set(title=f"cluster {cluster} ({n_series} items)");

    ax.set(xlabel="frequency (1/day)", ylabel="spectrum")

    

fig.tight_layout()
path = "clustered.csv"

df_unit.assign(cluster=clustered).to_csv(path, index=True)
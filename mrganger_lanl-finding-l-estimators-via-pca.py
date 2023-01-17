%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import librosa as lr

import librosa.display as lrd

np.warnings.filterwarnings('ignore')



def read(**kwargs):

    return pd.read_csv('../input/train.csv', **kwargs)



def spectrum(x, sr):

    Xr, Xi = lr.stft(np.real(x).astype(float)), lr.stft(np.imag(x).astype(float))

    Xdb = lr.amplitude_to_db(np.sqrt(abs(Xr)**2 + abs(Xi)**2))

    plt.figure(figsize=(14, 5))

    lrd.specshow(Xdb, sr=int(sr), x_axis='time', y_axis='hz')

    plt.show()

    

def waveplot(x, sr):

    xr, xi = np.real(x).copy(), np.imag(x).copy()

    plt.figure(figsize=(14, 5))

    lrd.waveplot(xr.astype(float), sr=int(sr))

    lrd.waveplot(xi.astype(float), sr=int(sr))

    plt.show()

    

x = read(nrows=1500000).acoustic_data.values

spectrum(x, 4e6)
from scipy.signal import firls, convolve, decimate

from scipy.spatial.distance import pdist



filt = firls(2001, bands=[0,240e3,245e3,250e3,255e3,2e6], desired=[0,0,1,1,0,0], fs=4e6)

waveplot(filt, 4e6);



def resample(xs):

    xs = convolve(xs.astype(float), filt, mode='valid')

    t = 2*np.pi*250e3/4e6*np.arange(len(xs))

    xs = xs*(np.cos(t) + 1j*np.sin(t))

    xs = decimate(xs, 150, ftype='fir')

    return xs



spectrum(resample(x), 4e6/150)

waveplot(resample(x), 4e6/150)
from tqdm import tqdm_notebook as tqdm

nrows = 629145481

chunksize = 150000



xs = []

ys = []

for df in tqdm(read(chunksize=chunksize), total=nrows//chunksize):

    xs += [resample(df.acoustic_data)]

    ys += [df.time_to_failure.iloc[-1]]
import dask

from dask.diagnostics import ProgressBar

from scipy.spatial.distance import pdist



def icdf(x):

    qs = np.linspace(0,1,200)

    return 2*np.quantile(pdist(np.column_stack([x.real,x.imag])), qs)



with ProgressBar():

    dists = np.array(dask.compute(*(dask.delayed(icdf)(xi) for xi in xs)))
from sklearn.decomposition import PCA

X_pre = dists

pca = PCA(n_components=20)

pca.fit(X_pre)

X = pca.transform(X_pre)



plt.figure(figsize=[15,15])

for i in range(10):

    comp = pca.components_[i]

    plt.plot(np.linspace(0,1,200), comp/abs(comp).max()/2+i)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor(n_neighbors=100, weights='distance')

X_post = X[:,:3]

plt.plot(np.sort(-cross_val_score(reg, X_post, ys, cv=8, n_jobs=4, scoring='neg_mean_absolute_error')));
reg.fit(X_post, ys)



@dask.delayed

def predict(fname):

    fdf = pd.read_csv(fname)

    x = fdf.acoustic_data.values

    x = resample(x)

    x = icdf(x)

    x = pca.transform(x[None,:])

    y = reg.predict(x[:,:3])

    return y[0]

    

import glob

files = pd.DataFrame(glob.glob('../input/test/*.csv'), columns=['filename'])

files['seg_id'] = files.filename.str.extract('.*/(seg_.*).csv', expand=True)

with ProgressBar():

    predictions = dask.compute(*(predict(fname) for fname in files.filename))

files['time_to_failure'] = predictions

files[['seg_id','time_to_failure']].to_csv('submission.csv', index=False)
!head submission.csv
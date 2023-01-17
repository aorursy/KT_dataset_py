import numpy as np

import pandas as pd

import os

import gc

import time

import numpy as np

import pandas as pd

from scipy import signal

from scipy import stats

from joblib import Parallel, delayed

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import seaborn as sns

from tsfresh.feature_extraction import feature_calculators

from scipy.signal import hilbert

import pywt 

from sklearn.cluster import DBSCAN

from statsmodels.robust import mad

%matplotlib inline
def maddest(d, axis=None):

    """

    Mean Absolute Deviation

    """

    

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)



def denoise_signal(x, wavelet='db4', level=1):

    """

    1. Adapted from waveletSmooth function found here:

    http://connor-johnson.com/2016/01/24/using-pywavelets-to-remove-high-frequency-noise/

    2. Threshold equation and using hard mode in threshold as mentioned

    in section '3.2 denoising based on optimized singular values' from paper by Tomas Vantuch:

    http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf

    """

    

    # Decompose to get the wavelet coefficients

    coeff = pywt.wavedec(x, wavelet, mode="per")

    

    # Calculate sigma for threshold as defined in http://dspace.vsb.cz/bitstream/handle/10084/133114/VAN431_FEI_P1807_1801V001_2018.pdf

    # As noted by @harshit92 MAD referred to in the paper is Mean Absolute Deviation not Median Absolute Deviation

    sigma = (1/0.6745) * maddest(coeff[-level])



    # Calculate the univeral threshold

    uthresh = sigma * np.sqrt(2*np.log(len(x)))

    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    

    # Reconstruct the signal using the thresholded coefficients

    return pywt.waverec(coeff, wavelet, mode='per')

class FeatureGenerator(object):

    def __init__(self, dtype, n_jobs=1, chunk_size=None):

        self.chunk_size = chunk_size

        self.dtype = dtype

        self.filename = None

        self.n_jobs = n_jobs

        self.test_files = []

        if self.dtype == 'train':

            self.filename = '../input/train.csv'

            self.total_data = int(629145481 / self.chunk_size)

        else:

            submission = pd.read_csv('../input/sample_submission.csv')

            for seg_id in submission.seg_id.values:

                self.test_files.append((seg_id, '../input/test/' + seg_id + '.csv'))

            self.total_data = int(len(submission))



    def read_chunks(self):

        if self.dtype == 'train':

            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,

                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})

            for counter, df in enumerate(iter_df):

                x = df.acoustic_data.values

                y = df.time_to_failure.values[-1]

                seg_id = 'train_' + str(counter)

                del df

                yield seg_id, x, y

        else:

            for seg_id, f in self.test_files:

                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})

                x = df.acoustic_data.values[-self.chunk_size:]

                del df

                yield seg_id, x, -999

    

    def get_features(self, x, y, seg_id):

        """

        Gets three groups of features: from original data and from reald and imaginary parts of FFT.

        """

        

        x = pd.Series(x)        

        main_dict = self.features(x, y, seg_id)

        

        return main_dict

        

    

    def features(self, x, y, seg_id):

        feature_dict = dict()

        feature_dict['target'] = y

        feature_dict['seg_id'] = seg_id

        x = pd.Series(denoise_signal(x, wavelet='db1', level=1))

        #x = x - np.mean(x)

    

        zc = np.fft.fft(x)

        zc = zc[:37500]



        # FFT transform values

        realFFT = np.real(zc)

        imagFFT = np.imag(zc)



        freq_bands = [x for x in range(0, 37500, 7500)]

        magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)

        phzFFT = np.arctan(imagFFT / realFFT)

        phzFFT[phzFFT == -np.inf] = -np.pi / 2.0

        phzFFT[phzFFT == np.inf] = np.pi / 2.0

        phzFFT = np.nan_to_num(phzFFT)



        for freq in freq_bands:

            if freq == 0:

                continue

            feature_dict['FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + 7500], 0.01)

            feature_dict['FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + 7500], 0.1)

            feature_dict['FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + 7500], 0.9)

            feature_dict['FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + 7500], 0.99)

            feature_dict['FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + 7500])

            feature_dict['FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + 7500])

            feature_dict['FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + 7500])

            

        for p in [10]:

            feature_dict[f'num_peaks_{p}'] = feature_calculators.number_peaks(x, 10)

            

        feature_dict['cid_ce'] = feature_calculators.cid_ce(x, normalize=True)

            

        for w in [5]:

            feature_dict[f'autocorrelation_{w}'] = feature_calculators.autocorrelation(x, w)

        return feature_dict



    def generate(self):

        feature_list = []

        res = Parallel(n_jobs=self.n_jobs,

                       backend='threading')(delayed(self.get_features)(x, y, s)

                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))

        for r in res:

            feature_list.append(r)

        return pd.DataFrame(feature_list)
training_fg = FeatureGenerator(dtype='train', n_jobs=-1, chunk_size=150000)

training_data = training_fg.generate()



test_fg = FeatureGenerator(dtype='test', n_jobs=-1, chunk_size=150000)

test_data = test_fg.generate()



X = training_data.drop(['target', 'seg_id'], axis=1)

X_test = test_data.drop(['target', 'seg_id'], axis=1)

test_segs = test_data.seg_id

y = training_data.target
del training_fg, training_data, test_fg, test_data; gc.collect()
mean_dict = {}

std_dict = {}

for col in X.columns:

    mean_value = X.loc[X[col] != -np.inf, col].mean()

    std_value = X.loc[X[col] != -np.inf, col].std()

    if X[col].isnull().any():

        print(col)

        X.loc[X[col] == -np.inf, col] = mean_value

        X[col] = X[col].fillna(mean_value)

    mean_dict[col] = mean_value

    std_dict[col] = std_value

    X[col] = (X[col] - mean_value)/std_value
for col in X_test.columns:

    if X_test[col].isnull().any():

        X_test.loc[X_test[col] == -np.inf, col] = mean_dict[col]

        X_test[col] = X_test[col].fillna(mean_dict[col])

    X_test[col] = (X_test[col] - mean_dict[col])/std_dict[col]
from sklearn.decomposition import PCA

pca = PCA(n_components=15)
def major_failures(x):

    return x > 45
x_pca = pca.fit_transform(X.values)

x_te_pca = pca.transform(X_test.values)



major_failure = major_failures(x_pca[:, 0])

major_failure_te = major_failures(x_te_pca[:, 0])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))



ax1.plot(x_pca[major_failure, 0], x_pca[major_failure, 1], '.r', alpha=0.75, label='major failure')

ax1.plot(x_pca[~major_failure, 0], x_pca[~major_failure, 1], '.', color='g', alpha=0.75, label='other time domain')

ax1.axvline(45, color='b')

ax1.set_title(f'Train PCA (Major Failures {sum(major_failure)})')

ax1.legend()



ax2.plot(x_te_pca[major_failure_te, 0], x_te_pca[major_failure_te, 1], '.r', alpha=0.75, label='major failure')

ax2.plot(x_te_pca[~major_failure_te, 0], x_te_pca[~major_failure_te, 1], '.', color='g', alpha=0.75, label='other time domain')

ax2.axvline(45, color='b')

ax2.set_title(f'Test PCA (Major Failures {sum(major_failure_te)})')

ax2.legend();
other_time_domain = ~major_failure

other_te_time_domain = ~major_failure_te



x_other = x_pca[other_time_domain]

x_te_other = x_te_pca[other_te_time_domain]
dbscan = DBSCAN(eps=1.5)

clust = dbscan.fit_predict(x_other[:, :3])

clust_te = dbscan.fit_predict(x_te_other[:, :3])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))



ax1.plot(x_other[clust==-1, 0], x_other[clust==-1, 1], '.', color='r', alpha=0.75, label='minor failure')

ax1.plot(x_other[clust!=-1, 0], x_other[clust!=-1, 1], '.', color='g', alpha=0.5, label='other time domain')

ax1.set_title(f'Train PCA (Minor Failures {sum(clust==-1)})')

ax1.legend()



ax2.plot(x_te_other[clust_te==-1, 0], x_te_other[clust_te==-1, 1], '.', color='r', alpha=0.75, label='minor failure')

ax2.plot(x_te_other[clust_te!=-1, 0], x_te_other[clust_te!=-1, 1], '.', color='g', alpha=0.5, label='other time domain')

ax2.set_title(f'Test PCA (Minor Failures {sum(clust_te==-1)})')

ax2.legend();
class UFailureStateDetector:

    def __init__(self, major_thr=45, minor_eps=1,n_components=5):

        self.major_thr = major_thr

        self.minor_eps = minor_eps

        self.n_components = n_components 

        self.pca = PCA(n_components=0.99)

        

    def fit_predict(self, x):      

        self.pca.fit(x)

        return self.predict(x)

    

    def predict(self, x):

        x = self.pca.transform(x)

        

        indexs = np.array([i for i in range(len(x))])

        major = self.major_failures(x, self.major_thr)

        major_indx = indexs[major]

        

        _x = x[~major]

        _indexs = indexs[~major]

        minor = self.minor_failures(_x, self.minor_eps, self.n_components)

        minor_indx = _indexs[minor]

        

        other_indx = np.array([i for i in indexs if i not in major_indx and i not in minor_indx])

        return other_indx, minor_indx, major_indx

        

    @staticmethod

    def major_failures(x, thr):

        return x[:, 0] > thr

    

    @staticmethod

    def minor_failures(x, eps, n_components):

        dbscan = DBSCAN(eps=eps)

        clust = dbscan.fit_predict(x[:, :n_components])

        minor = clust == -1

        return minor
ufsd = UFailureStateDetector(minor_eps=1.5)
other, minor, major = ufsd.fit_predict(X.values)

other_te, minor_te, major_te = ufsd.predict(X_test.values)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))



ax1.plot(x_pca[major, 0], x_pca[major, 1], '.r', alpha=0.75, label='major failure')

ax1.plot(x_pca[minor, 0], x_pca[minor, 1], '.b', alpha=0.5, label='minor failure')

ax1.plot(x_pca[other, 0], x_pca[other, 1], '.', color='g', alpha=0.35, label='other time domain')

ax1.axvline(45, color='b')

ax1.set_title(f'Train PCA (Major Failures - {len(major)}, Minor Failures - {len(minor)})')

ax1.legend()



ax2.plot(x_te_pca[major_te, 0], x_te_pca[major_te, 1], '.r', alpha=0.75, label='major failure')

ax2.plot(x_te_pca[minor_te, 0], x_te_pca[minor_te, 1], '.b', alpha=0.5, label='minor failure')

ax2.plot(x_te_pca[other_te, 0], x_te_pca[other_te, 1], '.', color='g', alpha=0.35, label='other time domain')

ax2.axvline(45, color='b')

ax2.set_title(f'Test PCA (Major Failures - {len(major_te)}, Minor Failures - {len(minor_te)})')

ax2.legend()

fig.savefig('pca_representation.png');
fig = plt.figure(figsize=(24, 8))

plt.plot(X.FFT_Mag_10q7500)



for m in major:

    plt.axvspan(m-5, m+5, color='r', alpha=0.5)

    

for m in minor:

    plt.axvspan(m-1, m+1, color='g', alpha=0.5)



fig.savefig('time_feature_representattion.png')
print('Minor in Train:', *minor, '\n')

print('Major in Train:', *major, '\n')
print('Minor in Test:', *test_segs.values[minor_te].tolist(), '\n')

print('Major in Train:', *test_segs.values[major_te].tolist(), '\n')
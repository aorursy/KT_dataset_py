# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import warnings

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

% matplotlib inline

warnings.filterwarnings('ignore')
def load_data(data, row, col):

    return np.asarray(data).reshape((row, col))



def svd_decompose(data):

    U, sigma, V = np.linalg.svd(data)

    sigma_norm = sigma / np.sum(sigma)

    #print('sigma is: ', sigma_norm)

    #print('cumulative sigma is', np.cumsum(sigma_norm))

    return U, sigma, V



def svd_recompose(data, K):

    U, sigma, V = svd_decompose(data)

    sig = np.zeros((K, K))

    for i in range(K):

        sig[i, i] = sigma[i]

    sig = np.mat(sig)

    recon = U[:, :K] * sig * V[:K, :]

    return recon.getA().reshape(-1)



def svd_plot(before, after):

    fig = plt.figure().add_subplot(111)

    fig.plot(np.asarray(before), 'b*', label='Original data')

    fig.plot(after, 'r-', label='Processed data')

    plt.show()
df=pd.read_csv('../input/test/seg_0a0fbb.csv')

svd_df=(svd_recompose(load_data(df, 300, 500), 5))

svd_plot(df, svd_df)
diff_df=abs(df).diff().fillna(0).values

svd_diff_df=(svd_recompose(load_data(diff_df, 300, 500), 5))

svd_plot(diff_df, svd_diff_df)
diff_df=pd.Series(diff_df.flatten())

diff_df2=abs(diff_df).diff().fillna(0).values

svd_diff_df2=(svd_recompose(load_data(diff_df2, 300, 500), 5))

svd_plot(diff_df2, svd_diff_df2)
def create_features(df, segment, seg_id, filter=True, window_range=[10]):

    x = pd.Series(segment['acoustic_data'].values)

    if filter == True:

        x = svd_recompose(load_data(x, 300, 500), 80)

        x = pd.Series(x)

    for windows in window_range:

        X_roll_kurtosis = x.rolling(windows).kurt().dropna().values

        df.loc[seg_id, 'q025_roll_kurt_' + str(windows)] = np.quantile(X_roll_kurtosis, 0.025)

        df.loc[seg_id, 'q050_roll_kurt_' + str(windows)] = np.quantile(X_roll_kurtosis, 0.05)

        df.loc[seg_id, 'q100_roll_kurt_' + str(windows)] = np.quantile(X_roll_kurtosis, 0.1)

        df.loc[seg_id, 'q400_roll_kurt_' + str(windows)] = np.quantile(X_roll_kurtosis, 0.4)

        

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})  #

rows = 150000

segments = int(np.floor(train.shape[0] / rows))

X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)

y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])



for segment in tqdm_notebook(range(segments)):

    seg = train.iloc[segment * rows:segment * rows + rows]

    y = seg['time_to_failure'].values[-1]

    y_tr.loc[segment, 'time_to_failure'] = y

    create_features(df=X_tr, segment=seg, seg_id=segment, filter=False)

tr = pd.concat([X_tr, y_tr], axis=1)

print((np.abs(tr.corrwith(tr['time_to_failure'])).sort_values(ascending=False)))
for segment in tqdm_notebook(range(segments)):

    seg = train.iloc[segment * rows:segment * rows + rows]

    y = seg['time_to_failure'].values[-1]

    y_tr.loc[segment, 'time_to_failure'] = y

    create_features(df=X_tr, segment=seg, seg_id=segment, filter=True)

tr = pd.concat([X_tr, y_tr], axis=1)

print((np.abs(tr.corrwith(tr['time_to_failure'])).sort_values(ascending=False)))
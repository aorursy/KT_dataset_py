import numpy as np
import pandas as pd
import glob

train = pd.read_csv('../input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
strain = []
for df in train:
    if len(df)==150_000:
        fullmean = df['acoustic_data'].mean()
        fullstd = df['acoustic_data'].std()
        fullmax = df['acoustic_data'].max()
        fullmin = df['acoustic_data'].min()
        lastmean = df['acoustic_data'][140_000:].mean()
        laststd = df['acoustic_data'][140_000:].std()
        lastmax = df['acoustic_data'][140_000:].max()
        lastmin = df['acoustic_data'][140_000:].min()
        lastTTF = df['time_to_failure'].values[-1]
        strain.append([fullmean, fullstd, fullmax, fullmin, lastmean, laststd, lastmax, lastmin, lastTTF])

strain = pd.DataFrame(strain, columns=['fullmean', 'fullstd', 'fullmax', 'fullmin', 'lastmean', 'laststd', 'lastmax', 'lastmin', 'time_to_failure'])
test = glob.glob('../input/test/**')
stest = []
for path in test:
    df = pd.read_csv(path, dtype={'acoustic_data': np.int16})
    seg_id = path.split('/')[-1].split('.')[0]
    fullmean = df['acoustic_data'].mean()
    fullstd = df['acoustic_data'].std()
    fullmax = df['acoustic_data'].max()
    fullmin = df['acoustic_data'].min()
    lastmean = df['acoustic_data'][140_000:].mean()
    laststd = df['acoustic_data'][140_000:].std()
    lastmax = df['acoustic_data'][140_000:].max()
    lastmin = df['acoustic_data'][140_000:].min()
    stest.append([seg_id, fullmean, fullstd, fullmax, fullmin, lastmean, laststd, lastmax, lastmin])
stest = pd.DataFrame(stest, columns=['seg_id', 'fullmean', 'fullstd', 'fullmax', 'fullmin', 'lastmean', 'laststd', 'lastmax', 'lastmin'])

sub = pd.read_csv('../input/sample_submission.csv')
strain.shape, stest.shape, sub.shape
col = [c for c in strain.columns if c not in ['time_to_failure']]

#https://www.kaggle.com/inversion/basic-feature-benchmark
from sklearn import *

scaler = preprocessing.StandardScaler()
scaled_train = scaler.fit_transform(strain[col])
scaled_test = scaler.transform(stest[col])

svm2 = svm.NuSVR(nu=0.6, C=1.3, kernel='rbf', gamma=10, tol=0.01)
svm2.fit(scaled_train, strain['time_to_failure'])
print(metrics.mean_absolute_error(strain['time_to_failure'], svm2.predict(scaled_train)))
stest['time_to_failure'] = svm2.predict(scaled_test)
stest[['seg_id','time_to_failure']].to_csv('submission.csv', index=False)
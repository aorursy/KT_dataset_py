import os

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler 

from tqdm import tqdm

from sklearn.svm import SVR,NuSVR

from sklearn.model_selection import GridSearchCV
train=pd.read_csv("../input/train.csv",dtype={"acoustic_data": np.int16, "time_to_failure": np.float64})
rows = 150000

segments = int(np.floor(train.shape[0] / rows))
col_names = ['mean','max','variance','min', 'stdev', 'q1', 'q5','q95', 'q99']
X1= pd.DataFrame(index=range(segments), dtype=np.float64, columns=col_names)

Y1 = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    Y1.loc[segment, 'time_to_failure'] = y

    X1.loc[segment, 'mean'] = x.mean()

    X1.loc[segment, 'stdev'] = x.std()

    X1.loc[segment, 'variance'] = np.var(x)

    X1.loc[segment, 'max'] = x.max()

    X1.loc[segment, 'min'] = x.min()

    X1.loc[segment, 'q1'] =  np.quantile(x, 0.01)

    X1.loc[segment, 'q5'] =  np.quantile(x, 0.05)



    X1.loc[segment, 'q95'] = np.quantile(x, 0.95)

    X1.loc[segment, 'q99'] = np.quantile(x, 0.99)  

    z = np.fft.fft(x)

    realFFT = np.real(z)

    imagFFT = np.imag(z)

    X1.loc[segment, 'A0'] = abs(z[0])

    X1.loc[segment, 'Real_mean'] = realFFT.mean()

    X1.loc[segment, 'Real_std'] = realFFT.std()

    X1.loc[segment, 'Real_max'] = realFFT.max()

    X1.loc[segment, 'Real_min'] = realFFT.min()

    X1.loc[segment, 'Imag_mean'] = imagFFT.mean()

    X1.loc[segment, 'Imag_std'] = imagFFT.std()

    X1.loc[segment, 'Imag_max'] = imagFFT.max()

    X1.loc[segment, 'Imag_min'] = imagFFT.min()
sc=StandardScaler()

sc.fit(X1)

scX = pd.DataFrame(sc.transform(X1), columns = X1.columns)
parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],'C': [0.1, 0.2, 0.5, 1, 1.5, 2]}]

model = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')

model.fit(scX, Y1.values.flatten())
sub=pd.read_csv("../input/sample_submission.csv",index_col='seg_id')

xtest=pd.DataFrame(columns=X1.columns,dtype=np.float64,index=sub.index)
for i, seg_id in enumerate(tqdm(xtest.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = pd.Series(seg['acoustic_data'].values)

    z = np.fft.fft(x)

    realFFT = np.real(z)

    imagFFT = np.imag(z)

    

    xtest.loc[seg_id, 'mean'] = x.mean()

    xtest.loc[seg_id, 'stdev'] = x.std()

    xtest.loc[seg_id, 'variance'] = np.var(x)

    xtest.loc[seg_id, 'max'] = x.max()

    xtest.loc[seg_id, 'min'] = x.min()

    xtest.loc[seg_id, 'q1'] = np.quantile(x, 0.01)

    xtest.loc[seg_id, 'q5'] = np.quantile(x, 0.05)

    

    xtest.loc[seg_id, 'q95'] = np.quantile(x, 0.95)

    xtest.loc[seg_id, 'q99'] = np.quantile(x, 0.99)

    xtest.loc[seg_id, 'A0'] = abs(z[0])

    xtest.loc[seg_id, 'Real_mean'] = realFFT.mean()

    xtest.loc[seg_id, 'Real_std'] = realFFT.std()

    xtest.loc[seg_id, 'Real_max'] = realFFT.max()

    xtest.loc[seg_id, 'Real_min'] = realFFT.min()

    xtest.loc[seg_id, 'Imag_mean'] = imagFFT.mean()

    xtest.loc[seg_id, 'Imag_std'] = imagFFT.std()

    xtest.loc[seg_id, 'Imag_max'] = imagFFT.max()

    xtest.loc[seg_id, 'Imag_min'] = imagFFT.min()
sctestx = pd.DataFrame(sc.transform(xtest), columns = xtest.columns)

pred = model.predict(sctestx)

print(pred.shape)
sub['time_to_failure'] = pred

sub.head()
sub.to_csv("submittedoutput.csv")
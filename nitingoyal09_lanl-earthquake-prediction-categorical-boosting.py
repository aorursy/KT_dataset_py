import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tqdm import tqdm

%matplotlib inline



from sklearn.preprocessing import StandardScaler # for data scaling

from sklearn.model_selection import GridSearchCV # hyperparameter optimization

from catboost import CatBoostRegressor, Pool #catagorical gradient boosting

from sklearn.svm import NuSVR, SVR



import os

IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/LANL/"

else:

    PATH="../input/"

os.listdir(PATH)

# Any results you write to the current directory are saved as output.
train = pd.read_csv(PATH+'train.csv', nrows = 6000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train.head(10)
#visualize the dataset

train_ad_sample_df = train['acoustic_data'].values[::100]

train_ttf_sample_df = train['time_to_failure'].values[::100]



def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title = "acoustic data + ttf"):

    fig, ax1 = plt.subplots(figsize = (12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color = 'r')

    ax1.set_ylabel('acoustic data', color='r')

    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color ='b')

    ax2.set_ylabel('time to failure', color = 'b')

    plt.legend(['time to faliure'], loc = (0.01, 0.9))

    plt.grid(True)

    

plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

del train_ad_sample_df

del train_ttf_sample_df
train = pd.read_csv(PATH+'train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

rows = 150000

segments = int(np.floor(train.shape[0] / rows))

print("Number of segments: ", segments)
features = ['mean','max','variance','min', 'stdev', 'quantile(0.01)', 'quantile(0.05)', 'quantile(0.95)', 'quantile(0.99)']



X = pd.DataFrame(index=range(segments), dtype=np.float64, columns=features)

Y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])



for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    Y.loc[segment, 'time_to_failure'] = y

    X.loc[segment, 'mean'] = x.mean()

    X.loc[segment, 'stdev'] = x.std()

    X.loc[segment, 'variance'] = np.var(x)

    X.loc[segment, 'max'] = x.max()

    X.loc[segment, 'min'] = x.min()

#     X.loc[segment, 'kur'] = x.kurtosis()

#     X.loc[segment, 'skew'] = x.skew()

    X.loc[segment, 'quantile(0.01)'] = np.quantile(x, 0.01)

    X.loc[segment, 'quantile(0.05)'] = np.quantile(x, 0.05)

    X.loc[segment, 'quantile(0.95)'] = np.quantile(x, 0.95)

    X.loc[segment, 'quantile(0.99)'] = np.quantile(x, 0.99)

    

    #FFT transform values -

    """

    from: 'https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction' kernel

     FFT is useful for sequence data feature extraction

     other than FFT there is Wavelet transform, which can be used to extract low level features, 

     Wavelet transform is though a lil bit complex in terms of computation

    """

    

    z = np.fft.fft(x)

    realFFT = np.real(z)

    imagFFT = np.imag(z)

    X.loc[segment, 'A0'] = abs(z[0])

    X.loc[segment, 'Rmean'] = realFFT.mean()

    X.loc[segment, 'Rstd'] = realFFT.std()

    X.loc[segment, 'Rmax'] = realFFT.max()

    X.loc[segment, 'Rmin'] = realFFT.min()

    X.loc[segment, 'Imean'] = imagFFT.mean()

    X.loc[segment, 'Istd'] = imagFFT.std()

    X.loc[segment, 'Imax'] = imagFFT.max()

    X.loc[segment, 'Imin'] = imagFFT.min()

    

X.describe().T
X.head()
# Scaling the data

scaler = StandardScaler()

scaler.fit(X)

scaled_X = pd.DataFrame(scaler.transform(X), columns = X.columns)
scaled_X.head(5)
# process the test data

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns = X.columns, dtype = np.float64, index = submission.index)

X_test.describe()
submission.shape, X_test.index.shape
# process the test data

for i, seg_id in enumerate(tqdm(X_test.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = pd.Series(seg['acoustic_data'].values)

    z = np.fft.fft(x)

    realFFT = np.real(z)

    imagFFT = np.imag(z)

    

    X_test.loc[seg_id, 'mean'] = x.mean()

    X_test.loc[seg_id, 'stdev'] = x.std()

    X_test.loc[seg_id, 'variance'] = np.var(x)

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()

    X_test.loc[seg_id, 'quantile(0.01)'] = np.quantile(x, 0.01)

    X_test.loc[seg_id, 'quantile(0.05)'] = np.quantile(x, 0.05)

    X_test.loc[seg_id, 'quantile(0.95)'] = np.quantile(x, 0.95)

    X_test.loc[seg_id, 'quantile(0.99)'] = np.quantile(x, 0.99)

    X_test.loc[seg_id, 'A0'] = abs(z[0])

    X_test.loc[seg_id, 'Rmean'] = realFFT.mean()

    X_test.loc[seg_id, 'Rstd'] = realFFT.std()

    X_test.loc[seg_id, 'Rmax'] = realFFT.max()

    X_test.loc[seg_id, 'Rmin'] = realFFT.min()

    X_test.loc[seg_id, 'Imean'] = imagFFT.mean()

    X_test.loc[seg_id, 'Istd'] = imagFFT.std()

    X_test.loc[seg_id, 'Imax'] = imagFFT.max()

    X_test.loc[seg_id, 'Imin'] = imagFFT.min()
# build a model

X_test.shape
# Scaling the test data

scaled_test_x = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

scaled_test_x.shape
scaled_test_x.tail()
train_pool = Pool(X, Y)

m = CatBoostRegressor(iterations = 10000, loss_function = 'MAE', boosting_type = 'Ordered')

m.fit(X, Y, silent = True)

m.best_score_
#predictions

predictions = np.zeros(len(scaled_test_x))

predictions += m.predict(scaled_test_x)
submission['time_to_failure'] = predictions
submission.head()
submission.to_csv('submission.csv')
parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

               'C': [0.1, 0.2, 0.5, 1, 1.5, 2]}]

reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')

reg1.fit(scaled_X, Y.values.flatten())
predictions = reg1.predict(scaled_test_x)

print(predictions.shape)
submission['time_to_failure'] = predictions

submission.head()
submission.to_csv('submissionSVM.csv')
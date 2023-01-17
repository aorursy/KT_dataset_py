# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error

import lightgbm as lgb



from keras.datasets import mnist

from keras.layers.core import Activation, Dense, Dropout

from keras.models import Sequential

from keras.optimizers import SGD

from keras.regularizers import l1, l2

from keras.utils import np_utils

import scipy.stats as scs

import scipy.signal as sig



import eli5

from eli5.sklearn import PermutationImportance

from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})


temp = train['acoustic_data'][:100]

temp_smooth = temp.rolling(10).mean()

print(len(sig.find_peaks(temp_smooth.values)[0]))



plt.plot(temp)

plt.plot(temp_smooth)

plt.show()
# rolling = train.acoustic_data.rolling(window=50).quantile(0.25)

# rolling[]

train.shape
rows = 150000

segments = int(np.floor(train.shape[0] / rows))

col_names = ['fft_{}'.format(i) for i in range(20)]

col_names = ['ave', 'std'] + col_names





X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=col_names)

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])

for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    seg_smooth = seg.rolling(10).mean()



    x = seg['acoustic_data'].values

    x_smooth = seg_smooth['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'abs_mean'] = np.absolute(x).mean()

    X_train.loc[segment, 'std'] = x.std()

#     X_train.loc[segment, 'max'] = x.max()

#     X_train.loc[segment, 'min'] = x.min()

    X_train.loc[segment, 'log_diff'] = np.log(x.ptp())

    

    X_train.loc[segment, 'kurt'] = scs.kurtosis(x)

    X_train.loc[segment, 'skew'] = scs.skew(x)

    

#     X_train.loc[segment, 'p25'] = np.percentile(x,0.25)

#     X_train.loc[segment, 'p50'] = np.percentile(x,0.50)

#     X_train.loc[segment, 'p75'] = np.percentile(x,0.75)

#     X_train.loc[segment, 'p80'] = np.percentile(x,0.80)

#     X_train.loc[segment, 'p85'] = np.percentile(x,0.85)

#     X_train.loc[segment, 'p90'] = np.percentile(x,0.90)

#     X_train.loc[segment, 'p95'] = np.percentile(x,0.95)

#     X_train.loc[segment, 'pinter'] = X_train.loc[segment, 'p75'] - X_train.loc[segment, 'p25']

    

    for i in range(0,10,2):

        temp_range = x[int(len(x)*i/10):int(len(x)*(i+2)/10)]

        temp_range_smooth = x_smooth[int(len(x)*i/10):int(len(x)*(i+2)/10)]

        X_train.loc[segment, f'ave{10*i}'] = temp_range.mean()

        X_train.loc[segment, f'std{10*i}'] = temp_range.std()

#         X_train.loc[segment, 'max{}'.format(10*i)] = x[int(len(x)*i/10):int(len(x)*(i+2)/10)].max()

#         X_train.loc[segment, 'min{}'.format(10*i)] = x[int(len(x)*i/10):int(len(x)*(i+2)/10)].min()

        X_train.loc[segment, f'log_diff{10*i}'] = np.log(temp_range.ptp())

        X_train.loc[segment, f'peaks_smooth{10*i}'] = len(sig.find_peaks(temp_range_smooth)[0])

        X_train.loc[segment, f'zeros_smooth{10*i}'] = ((temp_range_smooth[:-1] * temp_range_smooth[1:]) < 0).sum()

        X_train.loc[segment, f'peaks_diff_smooth{10*i}'] = np.mean(np.diff(sig.find_peaks(temp_range_smooth)[0]))

        

    freq = np.fft.fft(x, n=50).real

    freq_i = np.fft.fft(x, n=50).imag

    

    X_train.loc[segment, 'fftp80'] = np.percentile(freq,0.80)

    X_train.loc[segment, 'fftp85'] = np.percentile(freq,0.85)

    X_train.loc[segment, 'fftp90'] = np.percentile(freq,0.90)

    X_train.loc[segment, 'fftp95'] = np.percentile(freq,0.95)

    

    X_train.loc[segment, 'fftip80'] = np.percentile(freq_i,0.80)

    X_train.loc[segment, 'fftip85'] = np.percentile(freq_i,0.85)

    X_train.loc[segment, 'fftip90'] = np.percentile(freq_i,0.90)

    X_train.loc[segment, 'fftip95'] = np.percentile(freq_i,0.95)

    

    for idx, freq_val in enumerate(freq):

        X_train.loc[segment, f'fft_{idx}'] = freq_val

        

    for idx, freq_val in enumerate(freq_i):

        X_train.loc[segment, f'ffti_{idx}'] = freq_val

        

    X_train.loc[segment, 'peaks_smooth'] = len(sig.find_peaks(x_smooth)[0])

    X_train.loc[segment, 'zeros_smooth'] = ((x_smooth[:-1] * x_smooth[1:]) < 0).sum()

    

        
print(X_train.shape)

X_train.head()

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
svm = NuSVR(nu=0.5, verbose=True, kernel='rbf', degree=3, tol=0.00005, gamma='scale', shrinking=True)

svm.fit(X_train_scaled, y_train.values.flatten())

y_pred = svm.predict(X_train_scaled)
d_train = lgb.Dataset(X_train_scaled, label=y_train.values.flatten())

params = {'num_leaves': 128,

          'min_child_samples': 79,

          'objective': 'gamma',

          'max_depth': -1,

          'learning_rate': 0.01,

          "boosting_type": "gbdt",

          "subsample_freq": 5,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1302650970728192,

          'reg_lambda': 0.3603427518866501,

          'colsample_bytree': 0.1

         }



clf = lgb.train(params, d_train)

y_pred = clf.predict(X_train_scaled)
## Design a Deep Feedforward Neural Network and Its Training

## Several major considerations: network architecture, activation function, loss function, dropout, regularization.	

def DefineModel_FNN():

    ################################################################

    ## Network Structure

    first_layer_width = 10

    second_layer_width = 20

    ## Try more layers

    third_layer_width = 0

    forth_layer_width = 0

    fifth_layer_width = 0

    

    ################################################################

    ## Activation Function: relu, sigmoid, tanh, elu

    activation_func = 'relu' 

    # activation_func = 'sigmoid'

    # activation_func = 'tanh'

    ################################################################

    ################################################################    

    ## Loss Function:

    #

    loss_function = 'mae'

    # loss_function = 'mean_squared_error'

    ################################################################

    

    ################################################################    

    ## Dropout option

    #

    dropout_rate = 0.2

    # dropout_rate = 0.5

    # dropout_rate = 0.9

    ################################################################ 

    

    ################################################################    

    ## Regularization option

    #

    # weight_regularizer = l1(0.01)

#     weight_regularizer = l2(0.01)

    weight_regularizer = None

    ################################################################

    ################################################################    

    ## Learning Rate

    learning_rate = 0.025

    # learning_rate = 0.0001

    # learning_rate = 0.5

    ################################################################

    

    ## Initialize model.

    model = Sequential()

    ## First hidden layer with 'first_layer_width' neurons. 

    ## Also need to specify input dimension.

    ## 'Dense' means fully-connected.

    model.add( Dense(first_layer_width, input_dim=146, W_regularizer=weight_regularizer) )

    model.add( Activation(activation_func) )

    if dropout_rate > 0:

        model.add(Dropout(dropout_rate))

    ## Second hidden layer.

    if second_layer_width > 0:

        model.add( Dense(second_layer_width) )

        model.add( Activation(activation_func) )

        if dropout_rate > 0:

            model.add(Dropout(dropout_rate)) 

    

    ## Third hidden layer.

    if third_layer_width > 0:

        model.add( Dense(third_layer_width) )

        model.add( Activation(activation_func) )

        if dropout_rate > 0:

            model.add(Dropout(dropout_rate)) 

            

    ## forth hidden layer.

    if forth_layer_width > 0:

        model.add( Dense(forth_layer_width) )

        model.add( Activation(activation_func) )

        if dropout_rate > 0:

            model.add(Dropout(dropout_rate)) 

    

    ## fifth hidden layer.

    if fifth_layer_width > 0:

        model.add( Dense(fifth_layer_width) )

        model.add( Activation(activation_func) )

        if dropout_rate > 0:

            model.add( Dropout(dropout_rate) )

    

    ## Last layer has the same dimension as the number of classes

    model.add(Dense(1, activation='linear'))

    

    ## Define optimizer. In this tutorial/codelab, we select SGD.

    ## You can also use other methods, e.g., opt = RMSprop()

    opt = SGD(lr=learning_rate, clipnorm=5.)

    # opt = 'adam'

    

    ## Compile the model

    model.compile(loss=loss_function, optimizer=opt, metrics=['mse'])

    

    return model
X_train_scaled = scaler.transform(X_train)

# model = DefineModel_FNN()

model = KerasRegressor(build_fn=DefineModel_FNN)

model.fit(X_train_scaled, y_train.values.flatten(),epochs=300)

y_pred = model.predict(X_train_scaled)

perm = PermutationImportance(model, random_state=1).fit(X_train_scaled, y_train.values.flatten())
eli5.show_weights(perm, top=200, feature_names = X_train.columns.tolist())
plt.figure(figsize=(6, 6))

plt.scatter(y_train.values.flatten(), y_pred)

plt.xlim(0, 20)

plt.ylim(0, 20)

plt.xlabel('actual', fontsize=12)

plt.ylabel('predicted', fontsize=12)

plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

plt.show()
score = mean_absolute_error(y_train.values.flatten(), y_pred)

print(f'Score: {score:0.3f}')

# 2.277 w/ log_diff, var

# 2.270 w/ log_diff

# 2.207 w/ log_diff + last 10% data::fnn TURNED IN WITH SCORE 1.612

# 2.173 increased epochs, increased layer width, added one layer, lowered learning rate

# 2.143 doubled epochs

# 2.169 doubled batch size

# 2.493 lowered learning to 0.001

# 2.111 back to 0.025 lr, doubled epochs

# 2.108 added layer w/ 40 nodes

# 2.081 added 10 nodes to layer 1 TURNED IN WITH LOWER SCORE 1.631

# 2.121 Lowered epocks 

# 1.884 Added back in ptp, var, and fft features

# 1.594 fft increased from 20 to 50 features

# 0.065 Large network with 10% features added, total 113 TURNED IN WITH SCORE 2.017

# 1.903 Added droppout and reduced features to 88 TURNED IN WITH SCORE 1.747

# Same features with svm TURNED IN WITH LOWER SCORE NOPE

# 1.929 Added regularization, reduced layer size TURNED IN WITH SCORE 1.84

# 1.653 Increased fft to 100 TURNED IN WITH SCORE 1.913

# 1.444 50 fft real and 50 fft imag

# 1.380 reduced net size to 3 layers of 50, epochs to 300, and dropout to 0.2

# 2.341 lightgbm model using grandmaster parameters

# 1.230 Took out regularization TURNED IN WITH SCORE 1.902

# 1.813 Dropout to 0.3, reduced layer size to 30 30 30 TURNED IN WITH SCORE 1.951

# 1.609 Increased epochs to 500

# 1.614 Feature changes (more percentiles and took out variance and max/min) with weight information

# 1.580 Added fft and ffti percentiles COMMIT

# 1.771 Removed third layer

# 1.363 Added smoothed data peaks wow

# 1.485 Added zero crossings TURNED IN WITH SCORE 2.064

# 1.925 Same as above with SVM TURNED IN WITH SCORE 1.675

# 1.795 Two layers of size 20, 300 epochs

# 1.661 Removed percentile features and added rolling zero crosses TURNED IN WITH SCORE 1.831

# 1.919 Same as above with SVM 

# 1.910 Added avg distance between peaks

# 2.161 lightgbm TURNED IN WITH SCORE 1.919

# 1.769 Reduced network and dropout 

# 1.715 SVR with nu = 1

# 1.712 changed gamma to scale commited

# 2.178 nu = 0.2 committed

# 1.909 nu = 0.5 committed
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
for seg_id in X_test.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    seg_smooth = seg.rolling(10).mean()

    

    x = seg['acoustic_data'].values

    x_smooth = seg_smooth['acoustic_data'].values

    

    

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'abs_mean'] = np.absolute(x).mean()

    X_test.loc[seg_id, 'std'] = x.std()

#     X_test.loc[seg_id, 'max'] = x.max()

#     X_test.loc[seg_id, 'min'] = x.min()

    X_test.loc[seg_id, 'log_diff'] = np.log(x.max()-x.min())

    

    X_test.loc[seg_id, 'kurt'] = scs.kurtosis(x)

    X_test.loc[seg_id, 'skew'] = scs.skew(x)

    

#     X_test.loc[seg_id, 'p25'] = np.percentile(x,0.25)

#     X_test.loc[seg_id, 'p50'] = np.percentile(x,0.50)

#     X_test.loc[seg_id, 'p75'] = np.percentile(x,0.75)

#     X_test.loc[seg_id, 'p80'] = np.percentile(x,0.80)

#     X_test.loc[seg_id, 'p85'] = np.percentile(x,0.85)

#     X_test.loc[seg_id, 'p90'] = np.percentile(x,0.90)

#     X_test.loc[seg_id, 'p95'] = np.percentile(x,0.95)

#     X_test.loc[seg_id, 'pinter'] = X_test.loc[seg_id, 'p75'] - X_test.loc[seg_id, 'p25']

    

    for i in range(0,10,2):

        temp_x = x[int(len(x)*i/10):int(len(x)*(i+2)/10)]

        temp_x_smooth = x_smooth[int(len(x)*i/10):int(len(x)*(i+2)/10)]

        X_test.loc[seg_id, 'ave{}'.format(10*i)] = temp_x.mean()

        X_test.loc[seg_id, 'std{}'.format(10*i)] = temp_x.std()

#         X_test.loc[seg_id, 'max{}'.format(10*i)] = x[int(len(x)*i/10):int(len(x)*(i+2)/10)].max()

#         X_test.loc[seg_id, 'min{}'.format(10*i)] = x[int(len(x)*i/10):int(len(x)*(i+2)/10)].min()

        X_test.loc[seg_id, 'log_diff{}'.format(10*i)] = np.log(temp_x.max()-temp_x.min())

        X_test.loc[seg_id, f'peaks_smooth{10*i}'] = len(sig.find_peaks(temp_x_smooth)[0])

        X_test.loc[seg_id, f'zeros_smooth{10*i}'] = ((temp_x_smooth[:-1] * temp_x_smooth[1:]) < 0).sum()

        X_test.loc[seg_id, f'peaks_diff_smooth{10*i}'] = np.mean(np.diff(sig.find_peaks(temp_range_smooth)[0]))

    

    freq = np.fft.fft(x, n=50).real

    freq_i = np.fft.fft(x, n=50).imag

    

    

    X_test.loc[seg_id, 'fftp80'] = np.percentile(freq,0.80)

    X_test.loc[seg_id, 'fftp85'] = np.percentile(freq,0.85)

    X_test.loc[seg_id, 'fftp90'] = np.percentile(freq,0.90)

    X_test.loc[seg_id, 'fftp95'] = np.percentile(freq,0.95)

    

    X_test.loc[seg_id, 'fftip80'] = np.percentile(freq_i,0.80)

    X_test.loc[seg_id, 'fftip85'] = np.percentile(freq_i,0.85)

    X_test.loc[seg_id, 'fftip90'] = np.percentile(freq_i,0.90)

    X_test.loc[seg_id, 'fftip95'] = np.percentile(freq_i,0.95)

    

    for idx, freq_val in enumerate(freq):

        X_test.loc[seg_id, 'fft_{}'.format(idx)] = freq_val

        

    for idx, freq_val in enumerate(freq_i):

        X_test.loc[seg_id, 'ffti_{}'.format(idx)] = freq_val

    

    X_test.loc[seg_id, 'peaks_smooth'] = len(sig.find_peaks(x_smooth)[0])

    X_test.loc[seg_id, 'zeros_smooth'] = ((x_smooth[:-1] * x_smooth[1:]) < 0).sum()
X_test.head()
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled.size)

# print(X_test_scaled[:10])

# print(model.predict(X_test_scaled))

submission['time_to_failure'] = svm.predict(X_test_scaled)

submission.to_csv('submission.csv')
sub = pd.read_csv('submission.csv')

sub[:100]
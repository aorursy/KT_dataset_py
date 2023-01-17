# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.signal

import scipy.stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
%%time

train = pd.read_csv('../input/LANL-Earthquake-Prediction/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train.head()
%matplotlib inline



train_acoustic_data_small = train['acoustic_data'].values[::50]

train_time_to_failure_small = train['time_to_failure'].values[::50]



fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")

plt.plot(train_acoustic_data_small, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train_time_to_failure_small, color='g')

ax2.set_ylabel('time_to_failure', color='g')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)



del train_acoustic_data_small

del train_time_to_failure_small
%%time

feat_mat = np.zeros((15000,23))

fs = 4000000 # Hz

window_time = 0.0375 # seconds

offset = 0.01 # seconds

window_size = int(window_time*fs)



for i in np.arange(0,feat_mat.shape[0]):

    start = int(i*offset*fs)

    stop = int(window_size+i*offset*fs)

    seg = train.iloc[start:stop,0]



    feat_mat[i,0] = np.mean(seg)

    feat_mat[i,1] = np.var(seg)

    feat_mat[i,2] = scipy.stats.skew(seg)

    feat_mat[i,3] = scipy.stats.kurtosis(seg)

    feat_mat[i,-1] = train.iloc[stop,1]

    
lower_perc = np.percentile(train.iloc[:,0],np.arange(1,10))

upper_perc = np.percentile(train.iloc[:,0],np.arange(91,100))



for i in np.arange(0,feat_mat.shape[0]):

    start = int(i*offset*fs)

    stop = int(window_size+i*offset*fs)

    seg = train.iloc[start:stop,0]

    

    for j in np.arange(0,lower_perc.shape[0]):

        perc = np.size(np.where(seg>lower_perc[j])[0])

        feat_mat[i,4+j] = perc/seg.shape[0] 

        

    for j in np.arange(0,upper_perc.shape[0]):

        perc = np.size(np.where(seg>upper_perc[j])[0])

        feat_mat[i,13+j] = perc/seg.shape[0] 



    
df = pd.DataFrame(feat_mat,columns=['mean', 'var', 'skew', 'kurt','perc1','perc2','perc3','perc4','perc5','perc6','perc7','perc8','perc9','perc91','perc92','perc93','perc94','perc95','perc96','perc97','perc98','perc99','time_to_failure'],dtype=np.float64)
df.head()
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.preprocessing import scale

from sklearn.linear_model import LinearRegression, LassoCV, LassoLarsCV

from keras.models import Sequential

from keras.layers import Dense, LSTM

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor
dataset = df.values



X_train, X_test, y_train, y_test = train_test_split(dataset[:,:-1],dataset[:,-1],test_size=0.2)
lr = LinearRegression()

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)



print("MAE = ",mean_absolute_error(y_test,pred_lr))
# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)



# Train the model on training data

rf.fit(X_train, y_train)



pred_rf = rf.predict(X_test)

print("MAE = ",mean_absolute_error(y_test,pred_rf))
NN = MLPRegressor()

NN.fit(scale(X_train),y_train)



pred_nn = NN.predict(scale(X_test))



print("MAE = ", mean_absolute_error(y_test,pred_nn))
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

lcv = LassoCV()

llcv = LassoLarsCV()

lr = LinearRegression()

NN = MLPRegressor()



# Put as many of the models as you want in this list

clf_array = [rf,NN,lcv,llcv,lr]
for clf in clf_array:

    clf.fit(scale(X_train),y_train)

    pred = clf.predict(scale(X_test))

    vanilla_scores = mean_absolute_error(y_test,pred)

    

    bagging_clf = BaggingRegressor(clf, max_samples=0.25, max_features=1.0, random_state=27)

    bagging_clf.fit(scale(X_train),y_train)

    pred_bag = bagging_clf.predict(scale(X_test))

    bag_scores = mean_absolute_error(y_test,pred_bag)

    

    print("vanilla {}: {}",clf,vanilla_scores)

    print("bagging {}: {}",clf,bag_scores)



    
from tqdm import tqdm_notebook



submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})



X_test = pd.DataFrame(columns=df.columns, dtype=np.float64, index=submission.index)



for i, seg_id in enumerate(tqdm_notebook(X_test.index)):

    seg = pd.read_csv('../input/LANL-Earthquake-Prediction/test/' + seg_id + '.csv')

    X_test.loc[seg_id, 'mean'] = np.mean(seg.values)

    X_test.loc[seg_id, 'var'] = np.var(seg.values)

    X_test.loc[seg_id, 'skew'] = scipy.stats.skew(seg.values)

    X_test.loc[seg_id, 'kurt'] = scipy.stats.kurtosis(seg.values)

    

    for j in np.arange(0,9):

        perc = np.size(np.where(seg>lower_perc[j])[0])

        X_test.loc[seg_id, 'perc{}'.format(j+1)] = perc/seg.shape[0] 

        

        perc = np.size(np.where(seg>upper_perc[j])[0])

        X_test.loc[seg_id, 'perc9{}'.format(j+1)] = perc/seg.shape[0]



    
X_test.drop(columns=['time_to_failure'],inplace=True)

X_test.head()
# Pick any model to make predictions and place it where "rf" is currently.



rf.fit(scale(dataset[:,:-1]),dataset[:,-1])



predictions = rf.predict(scale(X_test))



submission['time_to_failure'] = predictions

submission.to_csv('submission.csv')
%matplotlib inline

import os

import warnings

warnings.simplefilter(action='ignore')



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import sklearn as sl

import scipy as sp



from tqdm import tqdm
train_data = pd.read_csv("/kaggle/input/collision-detection-ai-using-vibration-data/train_features.csv")

train_target = pd.read_csv("/kaggle/input/collision-detection-ai-using-vibration-data/train_target.csv")

test_data = pd.read_csv("/kaggle/input/collision-detection-ai-using-vibration-data/test_features.csv")
submission_file = pd.read_csv("/kaggle/input/collision-detection-ai-using-vibration-data/sample_submission.csv")
train_data.shape,train_target.shape
train_data.head()
train_target.head()
train_data.info()
train_data.id.nunique()
train_data[train_data.id == 0]
train_data[train_data.id == 1]
def plot_data(accelaration_df : pd.DataFrame,features : list, title : str) -> None:

    """ Plot the accelaration data

        :params accelaration_df: accelaration data for one id

        :params title: string

    """

    

    fig = plt.figure(figsize=(10,6))

    fig.tight_layout(pad=10.0)

    fig.suptitle(title)

    

    for idx,feature in enumerate(features):

        ax = fig.add_subplot(2,2,idx+1)

        accelaration_df[feature].plot(kind='line',

                                     title = title + " " + feature,

                                     ax=ax)
feats_to_plot = ["S1","S2","S3", "S4"]

plot_data(train_data[train_data.id == 0],feats_to_plot,"Accelaration Params")
train_target[train_target.id == 0]
feats_to_plot = ["S1","S2","S3", "S4"]

plot_data(train_data[train_data.id == 100],feats_to_plot,"Accelaration Params")
feats_to_plot = ["S1","S2","S3", "S4"]

plot_data(train_data[train_data.id == 250],feats_to_plot,"Accelaration Params")
feats_to_plot = ["S1","S2","S3", "S4"]

plot_data(train_data[train_data.id == 300],feats_to_plot,"Accelaration Params")
feats_to_plot = ["S1","S2","S3", "S4"]

plot_data(train_data[train_data.id == 400],feats_to_plot,"Accelaration Params")
fs = 5 #sampling frequency

fmax = 25 #sampling period

dt = 1/fs #length of signal

n = 75



def fft_features(data_set : pd.DataFrame) -> np.ndarray:

    """ Convert the dataset to fourier transfomed

        :params data_set: original collider params data

        :returns ft_data: Fourier transformed data

        #Reference - https://dacon.io/competitions/official/235614/codeshare/1174

    """

    ft_data = list()

    

    features = ["S1","S2","S3", "S4"]

    

    id_set = list(data_set.id.unique())

    

    for ids in tqdm(id_set):

        s1_fft = np.fft.fft(data_set[data_set.id==ids]['S1'].values)*dt

        s2_fft = np.fft.fft(data_set[data_set.id==ids]['S2'].values)*dt

        s3_fft = np.fft.fft(data_set[data_set.id==ids]['S3'].values)*dt

        s4_fft = np.fft.fft(data_set[data_set.id==ids]['S4'].values)*dt

        

        ft_data.append(np.concatenate([np.abs(s1_fft[0:int(n/2+1)]),

                                       np.abs(s2_fft[0:int(n/2+1)]),

                                       np.abs(s3_fft[0:int(n/2+1)]),

                                       np.abs(s4_fft[0:int(n/2+1)])]))

    

    return np.array(ft_data)
train_fft = fft_features(train_data)
train_fft.shape[0] == len(train_data.id.unique())
test_fft = fft_features(test_data)
test_fft.shape[0] == len(test_data.id.unique())
from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import GradientBoostingRegressor
base_model = GradientBoostingRegressor(loss='quantile',

                                      n_estimators=100,

                                      criterion='mae',

                                      random_state=2021,

                                      max_features='sqrt',

                                      n_iter_no_change=2)



mult_regressor = MultiOutputRegressor(base_model,

                                      n_jobs=-1)
mult_regressor.fit(train_fft,

                  train_target.drop(['id'],axis=1))
predictions = mult_regressor.predict(test_fft)
predictions[0]
submission_file[['X','Y','M','V']] = predictions

submission_file.head()
submission_file.to_csv("submission_1_1.csv",

                  index=False)
def generate_agg_feats(data_set : pd.DataFrame) -> pd.DataFrame:

    """ Create aggrage features from the data

        :param data_set: Base data as DataFrame

        :returns agg_data: Aggragated DataFrame

    """

    

    max_feats = data_set.groupby(['id']).max().add_suffix('_max').iloc[:,1:]

    min_feats = data_set.groupby(['id']).min().add_suffix('_min').iloc[:,1:]

    mean_feats = data_set.groupby(['id']).mean().add_suffix('_mean').iloc[:,1:]

    std_feats = data_set.groupby(['id']).std().add_suffix('_std').iloc[:,1:]

    median_feats = data_set.groupby(['id']).median().add_suffix('_median').iloc[:,1:]

    skew_feats = data_set.groupby(['id']).skew().add_suffix('_skew').iloc[:,1:]

    

    agg_data = pd.concat([max_feats,min_feats,

                          mean_feats,std_feats,median_feats,skew_feats],

                        axis=1)

    

    return agg_data
agg_train = generate_agg_feats(train_data)

agg_train.shape
agg_train.head()
agg_test = generate_agg_feats(test_data)

agg_test.shape
mult_regressor.fit(agg_train,

                  train_target.drop(['id'],axis=1))
agg_pred = mult_regressor.predict(agg_test)
agg_pred[0]
submission_file[['X','Y','M','V']] = agg_pred

submission_file.head()
submission_file.to_csv("submission_2.csv",

                  index=False)
from sklearn.svm import SVR

from sklearn.multioutput import RegressorChain
svr = SVR(kernel='rbf',

         gamma='auto',

         shrinking=True)

regressor_chain = RegressorChain(svr,

                                order='random',

                                random_state=1999)
regressor_chain.fit(agg_train,

                  train_target.drop(['id'],axis=1))
svr_p1 = regressor_chain.predict(agg_test)
submission_file[['X','Y','M','V']] = svr_p1

submission_file.head()
submission_file.to_csv("submission_3.csv",

                  index=False)
regressor_chain.fit(train_fft,

                  train_target.drop(['id'],axis=1))
fft_pred = regressor_chain.predict(test_fft)
submission_file[['X','Y','M','V']] = fft_pred

submission_file.head()
submission_file.to_csv("submission_4.csv",

                  index=False)
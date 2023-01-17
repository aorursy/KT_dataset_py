import numpy as np

import pandas as pd

from scipy.stats import norm

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import copy

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/train.csv')

x_data = df.drop(['ID_code','target'],axis=1).values

y_data = df['target'].values

df_test = pd.read_csv('../input/test.csv')

x_test = df_test.drop('ID_code',axis=1).values
np.max(np.triu(np.corrcoef(x_data,rowvar=False),k=1))
np.max(np.triu(np.corrcoef(x_data[y_data==1],rowvar=False),k=1))
def GetAucRoc(x_train, x_test, y_train, y_test,

              train_test_merger = False,

              n_bins = 23,

              range_limit = 3,

              bins_finding_method = 'Equal steps',

              n_vars = 200,

              smoothing_type=None,

              smoothing_epsilon=0):

    """

    This function returns ROC_AUC metrics of the submission provided by GetSubmission()

    function in the case if y_test is provided:

    y_test: test set of the targets (will be applied for measure the quality of submission)

    For more details see the GetSubmission() function

    """

    y_hat, y_hat_test = GetSubmission(x_train, x_test, y_train,

                                      train_test_merger = train_test_merger,

                                      n_bins = n_bins,

                                      range_limit = range_limit,

                                      bins_finding_method = bins_finding_method,

                                      n_vars = n_vars,

                                      smoothing_type = smoothing_type,

                                      smoothing_epsilon = smoothing_epsilon)

    res = list()

    res.append(roc_auc_score(y_train,y_hat))

    res.append(roc_auc_score(y_test,y_hat_test))

    return res



def GetSubmission(x_train, x_test, y_train,

              train_test_merger = False,

              n_bins = 23,

              range_limit = 3,

              bins_finding_method = 'Equal steps',

              n_vars = 200,

              smoothing_type = None,

              smoothing_epsilon = 0):

    """

    This function gets several parameters described below and returns the submission

    for Santander prediction competition based on the statisctical methods.

    x_train: train set of features

    x_test: test set of of features (that will be the base for prediction)

    y_train: train set of targets

    train_test_merger: if True the probability of feature being within the range will be

                       evaluated based on the data in both x_train and x_test, otherwise only

                       based ob the data in x_train

    n_bins: in order to calculate the probabilities the function divides the range of the feature

            values into segments, n_bins corresponds to number of such segments. Increasing of 

            the n_bins let make the prediction more accurate but with too big n_bins the 

            prediction becomes overfitted. If smoothing is applied the overfitting doesn't appear

            but there is no additional benefits from n_bins increasing

    range_limit: in order to calculate the probabilities the function divides the range of the

                 feature values into segments but the function also uses 2 open segments

                 (-infinity, m(var) - range_limit * std(var)) and 

                 (m(var) + range_limit * std(var), +infinity)

                 As a default the value 3 is applied that corresponds to 0,28% of the feature

                 values in this 2 open segments

    bins_finding_method: in order to divide the range of the feature values different methods can

                         be applied. In this function following of them are realized:

                         - "Equal normal probability" means that the range is divided into

                           segments so that if the feature is distributed normally than the 

                           probabilities that the feature value is within the different segments 

                           are equal

                         - "Linearly changing probability" means that the range is divided into

                           segments so that if the feature is distributed normally than the

                           probabilities that the feature value is within the different segments

                           increase linearly till the mean of distribution and than decrease 

                           linearly

                         - "Equal a posteriori probability" means that the range is divided into

                           segments so that each segments contains equal number of values from 

                           train set

                         - "Equal steps" is used as default method and means that the segments 

                           are of the equal length

                         Actually the difference between the methods is almost invisible 

                         especially if the smoothing is applied

    n_vars: number of vars, 200 for Santander competition

    smoothing_type: to avoid overfitting the smoothing can be applied. Smoothing can be realized 

                    in many different manners but only one type is realized in this function:

                    - 'plus_minus_epsilon' changes the calculation of probabilies. Without 

                      smoothing the probabilities are calculated as a frequency of the feature 

                      hitting within the range defined by bins, in the case of 

                      'plus_minus_epsilon' the range is enlarged by smoothing_epsilon from both 

                      sides

    smoothing_epsilon: see smoothing_type description

    """

    

    Scaler = StandardScaler()

    if train_test_merger:

        x_data = np.vstack((x_train,x_test))

    else:

        x_data = x_train.copy()

    x_data_scaled = Scaler.fit_transform(x_data)

    x_train_scaled = Scaler.transform(x_train)

    x_test_scaled = Scaler.transform(x_test)

    

    bins = list()

    if bins_finding_method == "Equal normal probability":

        for i in range(n_vars):

            norm_limit = norm.cdf(range_limit,loc=0,scale=1)

            bins.append(norm.isf(np.linspace(norm_limit, 1-norm_limit, n_bins+1),loc=0,scale=1))

    elif bins_finding_method == "Linearly changing probability":

        for i in range(n_vars):

            norm_limit = norm.cdf(range_limit,loc=0,scale=1)

            u = np.arange(1,n_bins//2+n_bins%2)

            u = np.hstack((u,np.flip(u)[n_bins%2:]))

            u_cum = np.cumsum(u)

            u_sum = u.sum()

            v = (2*norm_limit-1)*(1-np.r_[0,u_cum/u_sum])+(1-norm_limit)

            bins.append(norm.isf(v,loc=0,scale=1)) 

    elif bins_finding_method == "Equal a posteriori probability":

        for i in range(n_vars):

            r = np.zeros(n_bins+1)

            r[0] = - range_limit

            r[-1] = range_limit

            u = (np.floor(np.linspace(0,x_data_scaled[:,i].size,n_bins+1))[1:-1]).astype(int)

            v = x_data_scaled[:,i].copy()

            v.sort()

            r[1:-1] = np.minimum(np.maximum((v[u] + v[u+1]) / 2,-range_limit),range_limit)

            bins.append(r)

    else:

        for i in range(n_vars):

            bins.append(np.linspace(-range_limit, range_limit, n_bins+1))

            

    P_Vij = list()

    for i in range(n_vars):

        a0 = (x_data_scaled[:,i] < bins[i][0]).sum()

        if smoothing_type is None:

            a10 = (x_data_scaled[:,i].reshape(x_data_scaled[:,i].shape[0],1) >= bins[i][:-1])

            a11 = (x_data_scaled[:,i].reshape(x_data_scaled[:,i].shape[0],1) < bins[i][1:])

            a1 = (a10 & a11).sum(axis=0)

        elif smoothing_type == 'plus_minus_epsilon':

            a10 = (x_data_scaled[:,i].reshape(x_data_scaled[:,i].shape[0],1) >= bins[i][:-1]-smoothing_epsilon)

            a11 = (x_data_scaled[:,i].reshape(x_data_scaled[:,i].shape[0],1) < bins[i][1:]+smoothing_epsilon)

            a1 = (a10 & a11).sum(axis=0) * (bins[i][1:]-bins[i][:-1]) / (bins[i][1:]-bins[i][:-1]+2*smoothing_epsilon)            

        a2 = (x_data_scaled[:,i] >= bins[i][n_bins]).sum()

        P_Vij.append(np.r_[a0,a1,a2]/x_data_scaled[:,i].size)

        

    P_Vij_given_T = list()

    for i in range(n_vars):

        a0 = ((x_train_scaled[:,i] < bins[i][0]) * y_train).sum()

        if smoothing_type is None:

            a10 = (x_train_scaled[:,i].reshape(x_train_scaled[:,i].shape[0],1) >= bins[i][:-1])

            a11 = (x_train_scaled[:,i].reshape(x_train_scaled[:,i].shape[0],1) < bins[i][1:])

            a1 = ((a10 & a11)*y_train.reshape(y_train.shape[0],1)).sum(axis=0)

        elif smoothing_type == 'plus_minus_epsilon':

            a10 = (x_train_scaled[:,i].reshape(x_train_scaled[:,i].shape[0],1) >= bins[i][:-1]-smoothing_epsilon)

            a11 = (x_train_scaled[:,i].reshape(x_train_scaled[:,i].shape[0],1) < bins[i][1:]+smoothing_epsilon)

            a1 = ((a10 & a11)*y_train.reshape(y_train.shape[0],1)).sum(axis=0) * (bins[i][1:]-bins[i][:-1]) / (bins[i][1:]-bins[i][:-1]+2*smoothing_epsilon)            

        a2 = ((x_train_scaled[:,i] >= bins[i][n_bins]) * y_train).sum()

        P_Vij_given_T.append(np.r_[a0,a1,a2]/x_train_scaled[(y_train==1),i].size)

        

    P_T = y_train.sum() / y_train.size

    P_T_given_Vij = list()

    for i in range(n_vars):

        P_T_given_Vij.append(np.nan_to_num(P_Vij_given_T[i] * P_T / P_Vij[i]))

        

    y_hat = np.ones((x_train_scaled.shape[0],),dtype=float)

    y_hat = y_hat * P_T

    for i in range(n_vars):

        in_bin = n_bins + 1 - (x_train_scaled[:,i].reshape((x_train_scaled[:,i].shape[0],1)) < bins[i]).sum(axis=1)

        y_hat = y_hat * P_T_given_Vij[i][in_bin] / P_T

    y_hat_test = np.ones((x_test_scaled.shape[0],),dtype=float)

    y_hat_test = y_hat_test * P_T

    for i in range(n_vars):

        in_bin = n_bins + 1 - (x_test_scaled[:,i].reshape((x_test_scaled[:,i].shape[0],1)) < bins[i]).sum(axis=1)

        y_hat_test = y_hat_test * P_T_given_Vij[i][in_bin] / P_T

    res = list()

    res.append(y_hat)

    res.append(y_hat_test)

    return res
subm = GetSubmission(x_data, x_test, y_data,

                    train_test_merger=False,

                    n_bins=1000,

                    bins_finding_method='Equal steps',

                    smoothing_type = 'plus_minus_epsilon',

                    smoothing_epsilon = 0.3)
subm = pd.DataFrame({'ID_code':df_test['ID_code'], 'target':pd.Series(subm[1])})

subm.to_csv('submissionS.csv', index=False)
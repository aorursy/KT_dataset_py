# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import copy

import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, GridSearchCV

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

import os

from scipy.stats import norm, skew



import seaborn as sns

color = sns.color_palette()



print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub_df = pd.read_csv('../input/sample_submission.csv')
plt.plot(data['ID'],data['y'])
data = data[data.y < 250]
plt.plot(data['ID'],data['y'])
sns.distplot(data['y'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(data['y'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('target distribution')
from scipy import stats



fig = plt.figure()

res = stats.probplot(data['y'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

data["y"] = np.log1p(data["y"])



#Check the new distribution 

sns.distplot(data['y'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(data['y'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('target distribution')
from scipy import stats



fig = plt.figure()

res = stats.probplot(data['y'], plot=plt)

plt.show()
train_objs_num = len(data)

dataset = pd.concat(objs=[data, test], axis=0)

dataset = pd.get_dummies(dataset)

data = copy.copy(dataset[:train_objs_num])

test = copy.copy(dataset[train_objs_num:])
useless_var = data.mean()[data.mean() == 0].index

y_train = data.y

data.drop('y', axis = 1, inplace=True)

data.drop('ID', axis = 1, inplace= True)

data.drop(useless_var.values, axis = 1, inplace=True)



test.drop('y', axis = 1, inplace=True)

test.drop('ID', axis = 1, inplace= True)

test.drop(useless_var.values, axis = 1, inplace=True)
depth1= ['X314', 'X315', 'X118', 'X232', 'X136', 'X127', 'X189', 'X261', 'X47', 'X236', 'X5_n', 'X5_w', 'X383', 'X204', 'X339', 'X275', 'X152', 'X50', 'X1_f', 'X6_e', 'X5_r', 'X104', 'X5_aa', 'X240', 'X206', 'X241', 'X95', 'X1_l', 'X113', 'X5_ag', 'X8_s', 'X5_v', 'X100', 'X13', 'X8_k', 'X6_c', 'X223', 'X114', 'X19', 'X115', 'X1_u', 'X80', 'X4_c', 'X68', 'X168', 'X306', 'X61', 'X350', 'X300', 'X132', 'X342', 'X1_c', 'X6_l', 'X267', 'X379', 'X23', 'X316', 'X8_j', 'X6_g', 'X5_m', 'X12', 'X0_z', 'X0_x', 'X77', 'X354', 'X196', 'X8_a', 'X283', 'X51', 'X8_g', 'X225', 'X292', 'X2_r', 'X148', 'X8_d', 'X8_x', 'X203', 'X5_k', 'X8_o', 'X101', 'X220', 'X191', 'X5_ab', 'X8_i', 'X8_w', 'X56', 'X65', 'X0_ab', 'X131', 'X224', 'X155', 'X181', 'X351', 'X0_au', 'X64', 'X75', 'X5_af', 'X105', 'X6_j', 'X117', 'X151', 'X177', 'X0_l', 'X362', 'X3_g', 'X5_l', 'X8_v', 'X285', 'X378', 'X244', 'X343', 'X228', 'X5_ae', 'X5_p', 'X273', 'X380', 'X5_ah', 'X27', 'X1_t', 'X6_h', 'X345', 'X1_h', 'X176', 'X116', 'X171', 'X1_aa', 'X6_d', 'X201', 'X122', 'X3_a', 'X5_ad', 'X178', 'X143', 'X38', 'X142', 'X14', 'X361', 'X1_q']

depth2= [('X118', 'X314'), ('X314', 'X315'), ('X232', 'X314'), ('X136', 'X315'), ('X127', 'X232'), ('X189', 'X315'), ('X232', 'X315'), ('X314', 'X47'), ('X261', 'X47'), ('X315', 'X47'), ('X232', 'X261'), ('X113', 'X118'), ('X118', 'X261'), ('X261', 'X5_r'), ('X236', 'X47'), ('X314', 'X61'), ('X232', 'X5_k'), ('X115', 'X261'), ('X206', 'X50'), ('X236', 'X261'), ('X5_w', 'X8_k'), ('X383', 'X50'), ('X315', 'X6_c'), ('X232', 'X47'), ('X241', 'X5_n'), ('X47', 'X5_w'), ('X236', 'X275'), ('X1_l', 'X5_aa'), ('X127', 'X136'), ('X261', 'X61'), ('X152', 'X5_n'), ('X143', 'X315'), ('X118', 'X5_w'), ('X118', 'X5_r'), ('X5_aa', 'X5_w'), ('X104', 'X118'), ('X152', 'X383'), ('X275', 'X5_n'), ('X118', 'X236'), ('X383', 'X5_n'), ('X351', 'X47'), ('X5_aa', 'X80'), ('X113', 'X261'), ('X240', 'X6_l'), ('X236', 'X314'), ('X13', 'X1_l'), ('X316', 'X6_e'), ('X115', 'X1_f'), ('X275', 'X339'), ('X114', 'X1_c'), ('X136', 'X232'), ('X127', 'X6_e'), ('X339', 'X47'), ('X47', 'X61'), ('X115', 'X314'), ('X168', 'X8_k'), ('X1_u', 'X47'), ('X151', 'X236'), ('X127', 'X47'), ('X152', 'X19'), ('X223', 'X6_g'), ('X118', 'X1_aa'), ('X104', 'X68'), ('X101', 'X152'), ('X306', 'X5_w'), ('X23', 'X8_k'), ('X113', 'X314'), ('X5_r', 'X5_w'), ('X47', 'X5_n'), ('X383', 'X6_l'), ('X127', 'X5_n'), ('X204', 'X5_n'), ('X6_l', 'X8_j'), ('X0_x', 'X47'), ('X300', 'X383'), ('X240', 'X8_s'), ('X5_n', 'X5_w'), ('X1_c', 'X5_r'), ('X100', 'X342'), ('X196', 'X5_n'), ('X47', 'X5_aa'), ('X0_z', 'X152'), ('X47', 'X5_ad'), ('X204', 'X383'), ('X379', 'X5_m'), ('X105', 'X232'), ('X19', 'X5_n'), ('X118', 'X1_q'), ('X261', 'X339'), ('X191', 'X339'), ('X350', 'X47'), ('X236', 'X240'), ('X5_r', 'X8_v'), ('X148', 'X95'), ('X127', 'X339'), ('X350', 'X5_n'), ('X132', 'X283'), ('X77', 'X8_j'), ('X1_l', 'X56'), ('X0_x', 'X1_u'), ('X12', 'X5_ag'), ('X1_f', 'X8_d'), ('X204', 'X236'), ('X204', 'X316'), ('X116', 'X47'), ('X113', 'X8_a'), ('X6_l', 'X77'), ('X204', 'X95'), ('X225', 'X383'), ('X100', 'X5_ab'), ('X100', 'X1_f'), ('X5_m', 'X8_s'), ('X19', 'X6_g'), ('X115', 'X8_s'), ('X339', 'X383'), ('X127', 'X1_f'), ('X118', 'X316'), ('X204', 'X5_aa'), ('X118', 'X8_a'), ('X115', 'X155'), ('X1_f', 'X5_v'), ('X104', 'X275'), ('X132', 'X1_l'), ('X1_f', 'X3_g'), ('X1_c', 'X315'), ('X23', 'X5_w'), ('X132', 'X339'), ('X116', 'X77'), ('X12', 'X383'), ('X2_r', 'X8_a'), ('X131', 'X5_aa'), ('X51', 'X5_k'), ('X5_ah', 'X8_g'), ('X204', 'X4_c'), ('X383', 'X5_v'), ('X362', 'X5_n'), ('X220', 'X342'), ('X204', 'X339'), ('X132', 'X354'), ('X132', 'X203'), ('X204', 'X240'), ('X383', 'X5_l'), ('X236', 'X292'), ('X1_c', 'X343'), ('X267', 'X95'), ('X292', 'X378'), ('X350', 'X5_l'), ('X345', 'X61'), ('X240', 'X8_o'), ('X117', 'X275'), ('X306', 'X5_n'), ('X100', 'X8_o'), ('X0_x', 'X224'), ('X5_n', 'X6_j'), ('X178', 'X61'), ('X236', 'X383'), ('X19', 'X5_m'), ('X236', 'X95'), ('X132', 'X5_v'), ('X132', 'X6_g'), ('X132', 'X236'), ('X19', 'X204'), ('X5_ag', 'X5_k'), ('X315', 'X6_g'), ('X240', 'X8_x'), ('X267', 'X316'), ('X383', 'X75'), ('X261', 'X383'), ('X105', 'X5_ag'), ('X267', 'X383'), ('X5_v', 'X5_w'), ('X65', 'X6_j'), ('X177', 'X267'), ('X342', 'X95'), ('X354', 'X6_j'), ('X306', 'X75'), ('X12', 'X95'), ('X115', 'X64'), ('X105', 'X114'), ('X181', 'X316'), ('X350', 'X380'), ('X383', 'X5_r'), ('X383', 'X8_x'), ('X292', 'X5_n'), ('X113', 'X3_a'), ('X142', 'X68'), ('X0_x', 'X351'), ('X292', 'X339'), ('X342', 'X5_m'), ('X1_l', 'X8_w'), ('X204', 'X47'), ('X151', 'X6_g'), ('X6_h', 'X75'), ('X240', 'X342'), ('X1_h', 'X64'), ('X12', 'X5_v'), ('X267', 'X51'), ('X342', 'X5_ag'), ('X176', 'X8_s'), ('X339', 'X4_c'), ('X171', 'X64'), ('X339', 'X8_d'), ('X206', 'X383'), ('X285', 'X350'), ('X342', 'X5_v'), ('X14', 'X68'), ('X1_u', 'X203'), ('X148', 'X244'), ('X203', 'X4_c'), ('X228', 'X8_w'), ('X342', 'X5_aa'), ('X12', 'X5_r'), ('X0_au', 'X339'), ('X132', 'X8_i'), ('X203', 'X267'), ('X0_au', 'X0_z'), ('X100', 'X23'), ('X350', 'X8_v'), ('X292', 'X8_d'), ('X177', 'X1_u'), ('X4_c', 'X5_w'), ('X148', 'X267'), ('X104', 'X95'), ('X339', 'X95'), ('X1_t', 'X5_ae'), ('X206', 'X8_i'), ('X228', 'X5_aa'), ('X315', 'X351'), ('X104', 'X342'), ('X5_p', 'X8_a'), ('X0_l', 'X4_c'), ('X104', 'X206'), ('X5_v', 'X8_d'), ('X5_m', 'X95'), ('X316', 'X5_ah'), ('X104', 'X5_w'), ('X1_t', 'X8_a'), ('X292', 'X5_ag'), ('X292', 'X5_af'), ('X1_u', 'X6_d'), ('X306', 'X4_c'), ('X0_ab', 'X4_c'), ('X5_aa', 'X5_ah'), ('X240', 'X275'), ('X240', 'X95'), ('X342', 'X5_af'), ('X273', 'X5_ae'), ('X0_au', 'X5_r'), ('X115', 'X177'), ('X0_ab', 'X204')]

depth3=[('X136', 'X314', 'X315'), ('X127', 'X232', 'X314'), ('X189', 'X314', 'X315'), ('X232', 'X314', 'X315'), ('X113', 'X118', 'X314'), ('X314', 'X315', 'X47'), ('X143', 'X314', 'X315'), ('X232', 'X314', 'X47'), ('X314', 'X315', 'X6_c'), ('X118', 'X1_q', 'X314'), ('X118', 'X261', 'X5_r'), ('X236', 'X314', 'X47'), ('X118', 'X1_aa', 'X314'), ('X261', 'X315', 'X47'), ('X232', 'X261', 'X5_k'), ('X118', 'X314', 'X5_r'), ('X118', 'X314', 'X8_a'), ('X118', 'X1_aa', 'X261'), ('X261', 'X5_r', 'X8_v'), ('X127', 'X232', 'X261'), ('X314', 'X47', 'X61'), ('X236', 'X261', 'X275'), ('X206', 'X383', 'X50'), ('X178', 'X314', 'X61'), ('X127', 'X136', 'X232'), ('X115', 'X1_f', 'X261'), ('X236', 'X261', 'X47'), ('X115', 'X261', 'X8_s'), ('X104', 'X118', 'X5_w'), ('X136', 'X232', 'X261'), ('X118', 'X236', 'X5_w'), ('X168', 'X5_w', 'X8_k'), ('X261', 'X47', 'X5_ad'), ('X23', 'X5_w', 'X8_k'), ('X275', 'X383', 'X5_n'), ('X261', 'X47', 'X5_w'), ('X261', 'X47', 'X61'), ('X152', 'X19', 'X5_n'), ('X261', 'X345', 'X61'), ('X261', 'X5_r', 'X5_w'), ('X127', 'X232', 'X47'), ('X261', 'X350', 'X47'), ('X113', 'X314', 'X47'), ('X13', 'X1_l', 'X5_aa'), ('X127', 'X232', 'X6_e'), ('X0_x', 'X1_u', 'X47'), ('X113', 'X261', 'X8_a'), ('X0_x', 'X351', 'X47'), ('X232', 'X51', 'X5_k'), ('X306', 'X47', 'X5_w'), ('X315', 'X351', 'X47'), ('X241', 'X47', 'X5_n'), ('X204', 'X316', 'X6_e'), ('X115', 'X1_f', 'X314'), ('X232', 'X5_ag', 'X5_k'), ('X240', 'X383', 'X6_l'), ('X114', 'X1_c', 'X5_r'), ('X127', 'X232', 'X5_n'), ('X113', 'X261', 'X3_a'), ('X115', 'X155', 'X314'), ('X241', 'X275', 'X5_n'), ('X383', 'X6_l', 'X8_j'), ('X101', 'X152', 'X383'), ('X47', 'X5_n', 'X5_w'), ('X19', 'X223', 'X6_g'), ('X47', 'X5_w', 'X8_k'), ('X196', 'X275', 'X5_n'), ('X47', 'X5_aa', 'X80'), ('X191', 'X275', 'X339'), ('X0_x', 'X224', 'X47'), ('X1_l', 'X47', 'X5_aa'), ('X261', 'X339', 'X47'), ('X236', 'X240', 'X275'), ('X5_aa', 'X5_w', 'X80'), ('X300', 'X383', 'X5_l'), ('X0_z', 'X152', 'X19'), ('X127', 'X232', 'X339'), ('X148', 'X267', 'X95'), ('X1_l', 'X5_aa', 'X5_w'), ('X152', 'X204', 'X383'), ('X204', 'X350', 'X5_n'), ('X1_l', 'X56', 'X5_aa'), ('X105', 'X232', 'X5_ag'), ('X204', 'X236', 'X47'), ('X105', 'X114', 'X232'), ('X132', 'X283', 'X339'), ('X19', 'X204', 'X5_n'), ('X151', 'X236', 'X275'), ('X339', 'X383', 'X47'), ('X127', 'X1_f', 'X232'), ('X204', 'X5_aa', 'X5_w'), ('X100', 'X342', 'X5_ab'), ('X100', 'X1_f', 'X5_v'), ('X6_l', 'X77', 'X8_j'), ('X131', 'X204', 'X5_aa'), ('X12', 'X5_ag', 'X5_r'), ('X225', 'X383', 'X5_r'), ('X118', 'X204', 'X316'), ('X204', 'X383', 'X5_v'), ('X104', 'X275', 'X339'), ('X116', 'X351', 'X47'), ('X1_f', 'X236', 'X8_d'), ('X132', 'X1_l', 'X5_v'), ('X1_f', 'X3_g', 'X5_v'), ('X132', 'X339', 'X354'), ('X152', 'X19', 'X204'), ('X204', 'X383', 'X50'), ('X12', 'X383', 'X5_l'), ('X204', 'X339', 'X5_n'), ('X1_c', 'X315', 'X5_r'), ('X5_aa', 'X5_r', 'X5_w'), ('X151', 'X236', 'X47'), ('X116', 'X13', 'X47'), ('X117', 'X275', 'X339'), ('X116', 'X6_l', 'X77'), ('X306', 'X362', 'X5_n'), ('X1_t', 'X2_r', 'X8_a'), ('X236', 'X383', 'X95'), ('X100', 'X342', 'X8_o'), ('X220', 'X342', 'X5_aa'), ('X19', 'X204', 'X5_m'), ('X5_aa', 'X5_ah', 'X8_g'), ('X132', 'X236', 'X5_v'), ('X1_u', 'X47', 'X95'), ('X100', 'X240', 'X342'), ('X1_c', 'X343', 'X5_r'), ('X5_aa', 'X5_w', 'X8_k'), ('X379', 'X5_m', 'X95'), ('X267', 'X316', 'X383'), ('X240', 'X6_l', 'X8_o'), ('X306', 'X5_n', 'X6_j'), ('X12', 'X342', 'X95'), ('X350', 'X380', 'X5_l'), ('X5_aa', 'X5_v', 'X5_w'), ('X261', 'X383', 'X5_r'), ('X204', 'X383', 'X8_x'), ('X19', 'X315', 'X6_g'), ('X306', 'X383', 'X75'), ('X292', 'X339', 'X5_n'), ('X240', 'X6_l', 'X8_x'), ('X1_l', 'X5_aa', 'X8_w'), ('X267', 'X383', 'X51'), ('X240', 'X342', 'X5_m'), ('X1_h', 'X241', 'X5_n'), ('X0_l', 'X379', 'X5_m'), ('X204', 'X47', 'X95'), ('X177', 'X1_u', 'X267'), ('X132', 'X203', 'X6_g'), ('X23', 'X5_aa', 'X5_w'), ('X181', 'X204', 'X316'), ('X176', 'X240', 'X8_s'), ('X104', 'X142', 'X68'), ('X306', 'X6_h', 'X75'), ('X339', 'X4_c', 'X8_d'), ('X0_ab', 'X65', 'X6_j'), ('X0_ab', 'X354', 'X6_j'), ('X115', 'X1_h', 'X64'), ('X204', 'X240', 'X95'), ('X5_m', 'X8_s', 'X95'), ('X12', 'X5_r', 'X5_v'), ('X104', 'X5_w', 'X68'), ('X132', 'X203', 'X4_c'), ('X151', 'X19', 'X6_g'), ('X292', 'X339', 'X378'), ('X342', 'X5_aa', 'X5_v'), ('X148', 'X244', 'X95'), ('X115', 'X177', 'X267'), ('X203', 'X267', 'X4_c'), ('X151', 'X236', 'X383'), ('X0_au', 'X0_z', 'X339'), ('X104', 'X206', 'X383'), ('X240', 'X8_s', 'X95'), ('X285', 'X350', 'X5_l'), ('X1_u', 'X203', 'X47'), ('X104', 'X14', 'X68'), ('X342', 'X4_c', 'X5_ag'), ('X171', 'X1_h', 'X64'), ('X204', 'X4_c', 'X5_w'), ('X228', 'X5_aa', 'X8_w'), ('X100', 'X23', 'X5_w'), ('X236', 'X292', 'X8_d'), ('X104', 'X339', 'X95'), ('X204', 'X240', 'X8_s'), ('X132', 'X203', 'X8_i'), ('X0_l', 'X5_m', 'X8_s'), ('X0_l', 'X204', 'X4_c'), ('X350', 'X5_l', 'X8_v'), ('X104', 'X206', 'X8_i'), ('X104', 'X342', 'X5_w'), ('X1_t', 'X5_p', 'X8_a'), ('X0_au', 'X273', 'X339'), ('X236', 'X292', 'X5_ag'), ('X316', 'X5_aa', 'X5_ah'), ('X1_u', 'X47', 'X6_d'), ('X0_ab', 'X306', 'X4_c'), ('X236', 'X5_v', 'X8_d'), ('X236', 'X292', 'X5_af'), ('X240', 'X275', 'X95'), ('X1_t', 'X273', 'X5_ae'), ('X0_au', 'X4_c', 'X5_r'), ('X0_ab', 'X204', 'X5_ag'), ('X342', 'X4_c', 'X5_af'), ('X122', 'X1_t', 'X5_ae'), ('X19', 'X1_h', 'X5_n'), ('X236', 'X292', 'X378'), ('X204', 'X240', 'X261'), ('X114', 'X228', 'X5_aa'), ('X177', 'X203', 'X27'), ('X0_au', 'X201', 'X4_c'), ('X203', 'X27', 'X8_g'), ('X0_l', 'X131', 'X38'), ('X0_l', 'X131', 'X361')]
data = data[depth1]

test = test[depth1]

for (a,b) in depth2[0:16]:

    data.loc[:,a+'_+_'+b] = data[a].add(data[b])

    data.loc[:,a+'_*_'+b] = data[a].mul(data[b])

    #data.loc[:,a+'_-_'+b] = data[a].sub(data[b])

    data.loc[:,a+'_abs(-)_'+b] = np.abs(data[a].sub(data[b]))

    

    test.loc[:,a+'_+_'+b] = test[a].add(test[b])

    test.loc[:,a+'_*_'+b] = test[a].mul(test[b])

    #test.loc[:,a+'_-_'+b] = test[a].sub(test[b])

    test.loc[:,a+'_abs(-)_'+b] = np.abs(test[a].sub(test[b]))

    #data.loc[:,a+'_max_'+b] = np.maximum(data[a],data[b])

    #data.loc[:,a+'_min_'+b] = np.minimum(data[a],data[b])

    

for (a,b,c) in depth3[0:12]:

    data.loc[:,a+'_+_'+b+'_+_'+c] = data[a].add(data[b]).add(data[c])

    test.loc[:,a+'_+_'+b+'_+_'+c] = test[a].add(test[b]).add(test[c])

    data.loc[:,a+'_*_'+b+'_*_'+c] = data[a].mul(data[b]).mul(data[c])

    test.loc[:,a+'_*_'+b+'_*_'+c] = test[a].mul(test[b]).mul(test[c])

    data.loc[:,a+'_+_'+b+'_*_'+c] = data[a].add(data[b]).mul(data[c])

    test.loc[:,a+'_+_'+b+'_*_'+c] = test[a].add(test[b]).mul(test[c])

    data.loc[:,a+'_*_'+b+'_+_'+c] = data[a].mul(data[b]).add(data[c])

    test.loc[:,a+'_*_'+b+'_+_'+c] = test[a].mul(test[b]).add(test[c])

    

    test.loc[:,a+'_-_'+b+'_+_'+c] = np.abs(test[a].sub(test[b])).add(test[c])

    test.loc[:,a+'_-_'+b+'_*_'+c] = np.abs(test[a].sub(test[b])).mul(test[c])

    test.loc[:,a+'_*_'+b+'_-_'+c] = test[a].mul(np.abs((test[b]).sub(test[c])))

    test.loc[:,a+'_+_'+b+'_-_'+c] = test[a].add(np.abs((test[b]).sub(test[c])))

    test.loc[:,a+'_-_'+b+'_-_'+c] = np.abs(np.abs(test[a].sub(test[b])).sub(test[c]))

    

    data.loc[:,a+'_-_'+b+'_+_'+c] = np.abs(data[a].sub(data[b])).add(data[c])

    data.loc[:,a+'_-_'+b+'_*_'+c] = np.abs(data[a].sub(data[b])).mul(data[c])

    data.loc[:,a+'_*_'+b+'_-_'+c] = data[a].mul(np.abs((data[b]).sub(data[c])))

    data.loc[:,a+'_+_'+b+'_-_'+c] = data[a].add(np.abs((data[b]).sub(data[c])))

    data.loc[:,a+'_-_'+b+'_-_'+c] = np.abs(np.abs(data[a].sub(data[b])).sub(data[c]))
from  itertools import combinations

from sklearn.metrics import matthews_corrcoef

cc = list(combinations(data.columns,2))

data_1 = [(c[0],c[1],abs(matthews_corrcoef(data[c[1]],data[c[0]]))) for c in cc]
#to_drop = []

#for (c1,c2,score) in data_1:

#    if score >0.95:

#        to_drop.append(c2)

#print(to_drop)

#data.drop(to_drop, axis=1)

#test.drop(to_drop, axis=1)
data.shape
from sklearn.model_selection import KFold, cross_val_score

n_folds = 5



def r2_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(data.values)

    r2= cross_val_score(model, data.values, y_train, scoring="r2", cv = kf)

    return(r2)
model_xgb = xgb.XGBRegressor(alpha= 0, colsample_bytree= 1, eta= 0.05, reg_lambda= 0.8, max_depth= 4, min_child_weight= 0.8, n_estimators= 100)
#XGB Parameter Tuning

gridParams = {

    'eta': [0.05, 0.1, 0.3],

    'max_depth': [2,4,6,9],

    'n_estimators': [100,400],

    'min_child_weight' : [0.8,1,1.2],

    'colsample_bytree' : [0.3,0.6,1],

    'lambda' : [0.8,1,1.2],

    'alpha': [0,0.5]

    }

#grid = GridSearchCV(model_xgb, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth= 6, min_samples_leaf= 12, min_samples_split= 6, n_estimators= 100)
#RF Parameter Tuning

gridParams = {

    'max_depth': [2,4,6,8,12],

    'n_estimators': [10,100],

    'min_samples_leaf' : [2,6,10],

    'min_samples_split' : [2,4,6]

    }

#grid = GridSearchCV(rf, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
model_lgb = lgb.LGBMRegressor(objective='regression', learning_rate=0.05, feature_fraction= 0.8, lambda_l1= 0, max_depth= 6, min_data_in_leaf= 50, n_estimators= 100, num_leaves= 8)
#lgb Parameter Tuning

gridParams = {

    'num_leaves': [4,6,8],

    'max_depth': [2,3,6],

    'n_estimators': [100,500,1000],

    'feature_fraction' : [0.2,0.6,1],

    'lambda_l1' : [0.5,0.6],

    'min_data_in_leaf': [6,12,20]

    }

#grid = GridSearchCV(model_lgb, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from sklearn.kernel_ridge import KernelRidge

KRR = KernelRidge(kernel='polynomial', alpha = 1, coef0=3, degree = 2)
#KRR Parameter Tuning

gridParams = {

    'alpha': [1,2,3],

    'degree': [2,3,4],

    'coef0' : [1,2,3],

}

#grid = GridSearchCV(KRR, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from sklearn.svm import SVR

svr = SVR(C= 1, gamma= 0.001)
#SVR Parameter Tuning

gridParams = {

    'gamma': [0.001,0.01,0.1],

    'C': [1,10,100],

}

#grid = GridSearchCV(svr, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from sklearn.linear_model import Lasso

lasso = Lasso(alpha =0.0005, random_state=1)
#Lasso Parameter Tuning

gridParams = {

    'alpha': [0.0005,0.005,0.05,0.5,1],

}

#grid = GridSearchCV(lasso, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from sklearn.ensemble import ExtraTreesRegressor

et = ExtraTreesRegressor(max_depth= 6, min_samples_leaf= 40, n_estimators= 500)
#Extra Trees Parameter Tuning

gridParams = {

    'n_estimators': [500],

    'max_depth': [6,8,10],

    'min_samples_leaf' : [40,60,80],

    #'min_samples_split': [0.2,0.6,0.9],

}

#grid = GridSearchCV(et, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(learning_rate= 0.005, n_estimators= 50)
#Adaboost Parameter Tuning

gridParams = {

    'n_estimators': [30,40,50],

    'learning_rate': [0.005, 0.05,0.5,1]

}

#grid = GridSearchCV(ada, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from catboost import CatBoostRegressor

cat = CatBoostRegressor(iterations=3000,learning_rate=0.005,depth=2)
#CatBoost Parameter Tuning

gridParams = {

            'iterations': [2500,3000],

            'learning_rate': [0.005],

            'depth' : [2]

}

#grid = GridSearchCV(cat, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.005,max_depth=3,alpha=0.5)
#GradientBoostingRegressor Parameter Tuning

gridParams = {

            'n_estimators': [1000],

            'learning_rate': [0.005],

            'max_depth' : [3],

            'alpha':[0.5]

    

}

#grid = GridSearchCV(gbr, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
from sklearn.linear_model import ElasticNet

ela = ElasticNet(alpha=0,l1_ratio=0,max_iter=5)
#ElasticNEt Parameter Tuning

gridParams = {

            'alpha': [0],

            'max_iter':[0,5,10,15],

            'l1_ratio': [0]

}

#grid = GridSearchCV(ela, gridParams,verbose=1, cv=4,n_jobs=2,scoring = 'r2')

# Run the grid

#grid.fit(data, y_train)



# Print the best parameters found

#print(grid.best_params_)

#print(grid.best_score_)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (KRR, model_lgb, rf, svr, model_xgb, lasso, ela,ada, et, gbr, cat))



#score = r2_cv(averaged_models)

#print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_models.fit(data.values,y_train)
predictions = np.expm1(averaged_models.predict(test.values))
sub_df.drop('y', axis = 1, inplace = True)

sub_df['y'] = predictions

sub_df.to_csv("submission.csv", index=False)

sub_df
import numpy as np

import time

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import accuracy_score, r2_score

from sklearn.decomposition import PCA

from sklearn import preprocessing as pp

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

import os
data = pd.read_hdf('../input/cleandata/clean.h5')

data.reset_index(inplace=True, drop=True)

# data.to_csv('clean.csv', index=False, sep=' ')

lc = data.columns.tolist()

del lc[14]

lc.append('sulfate_dose')

data = data[lc]

data.to_csv('clean.csv', index=False, sep=' ')

# data.to_excel('clean.xlsx')

X = data.drop('sulfate_dose', axis=1)

Y = data.loc[:, ['sulfate_dose']]
from sklearn import preprocessing

names = X.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_scaled = scaled_df





from sklearn import preprocessing as pp

poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_scaled

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)



data_poly = pd.concat([output_df, Y], axis=1)

data_poly.to_csv('clean.csv', index=False, sep=' ')
data = pd.read_hdf('../input/cleandata/clean.h5')

data.reset_index(inplace=True, drop=True)

# data.to_csv('clean.csv', index=False, sep=' ')

lc = data.columns.tolist()

del lc[14]

lc.append('sulfate_dose')

data = data[lc]



Y_traite = data.loc[:, 'ph_s1':'cl_s1'] # les target de ph_s1 à cl_s1

X_traite = data.drop(['ph_s1', 't_s1', 'cond_s1', 'turb_s1', 'cl_s1'], axis=1) # les variables data sans y_traite avec drop

data_41_ph = pd.concat([X_traite, Y_traite.loc[:, 'ph_s1']], axis=1) # ph_s1

data_41_t = pd.concat([X_traite, Y_traite.loc[:, 't_s1']], axis=1) # t_s1

data_41_cond = pd.concat([X_traite, Y_traite.loc[:, 'cond_s1']], axis=1) # cond_s1

data_41_turb = pd.concat([X_traite, Y_traite.loc[:, 'turb_s1']], axis=1) # turb_s1

data_41_cl = pd.concat([X_traite, Y_traite.loc[:, 'cl_s1']], axis=1) # cl_s1

from sklearn import preprocessing

names = X_traite.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X_traite)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_scaled = scaled_df





from sklearn import preprocessing as pp

poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_scaled

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)



data_poly = pd.concat([output_df, Y_traite.loc[:,['ph_s1']]], axis=1)

data_poly.to_csv('data_41_ph.csv', index=False, sep=' ')

from sklearn import preprocessing

names = X_traite.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X_traite)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_scaled = scaled_df





from sklearn import preprocessing as pp

poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_scaled

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)



data_poly = pd.concat([output_df, Y_traite.loc[:,['cond_s1']]], axis=1)

data_poly.to_csv('data_41_cond.csv', index=False, sep=' ')
from sklearn import preprocessing

names = X_traite.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X_traite)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_scaled = scaled_df





from sklearn import preprocessing as pp

poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_scaled

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)



data_poly = pd.concat([output_df, Y_traite.loc[:,['t_s1']]], axis=1)

data_poly.to_csv('data_41_t.csv', index=False, sep=' ')
from sklearn import preprocessing

names = X_traite.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X_traite)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_scaled = scaled_df





from sklearn import preprocessing as pp

poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_scaled

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)



data_poly = pd.concat([output_df, Y_traite.loc[:,['turb_s1']]], axis=1)

data_poly.to_csv('data_41_turb.csv', index=False, sep=' ')
from sklearn import preprocessing

names = X_traite.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(X_traite)

scaled_df = pd.DataFrame(scaled_df, columns=names)

X_scaled = scaled_df





from sklearn import preprocessing as pp

poly = pp.PolynomialFeatures(2, include_bias=False)

input_df = X_scaled

output_nparray = poly.fit_transform(input_df)

target_feature_names = ['*'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(input_df.columns,p) for p in poly.powers_]]

output_df = pd.DataFrame(output_nparray, columns = target_feature_names)



data_poly = pd.concat([output_df, Y_traite.loc[:,['cl_s1']]], axis=1)

data_poly.to_csv('data_41_cl.csv', index=False, sep=' ')
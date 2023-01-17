import pandas as pd

import numpy as np
import scipy

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np

import random

import datetime

import warnings

warnings.filterwarnings('ignore')

from scipy import stats



# Imports for better visualization

from matplotlib import rcParams

#colorbrewer2 Dark2 qualitative color table

dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),

                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),

                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),

                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),

                (0.4, 0.6509803921568628, 0.11764705882352941),

                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),

                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]



rcParams['figure.figsize'] = (10, 4)

rcParams['figure.dpi'] = 150

#rcParams['axes.color_cycle'] = dark2_colors

rcParams['lines.linewidth'] = 2

rcParams['font.size'] = 8

rcParams['patch.edgecolor'] = 'white'

rcParams['patch.facecolor'] = dark2_colors[0]

rcParams['font.family'] = 'StixGeneral'

rcParams['axes.grid'] = True

rcParams['axes.facecolor'] = '#eeeeee'
from subprocess import check_output

print(check_output(["ls", "../input/"]).decode("utf8"))
train = pd.read_csv('../input/astro-analytics-techsoc-iitm/train_techsoc.csv')

test = pd.read_csv('../input/astro-analytics-techsoc-iitm/test_techsoc.csv')

sample = pd.read_csv('../input/astro-analytics-techsoc-iitm/sample_submission_techsoc.csv')

train.shape, test.shape, sample.shape
uniq = list(set(test['sat_id']).intersection(train['sat_id']))

filter_train = pd.DataFrame()

for i in uniq:

    df = train[train['sat_id']==i].reset_index(drop=True)

    filter_train = filter_train.append(df)

train_df = filter_train.reset_index(drop=True)

train_df.head()
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
%%time

pred_x = pd.DataFrame()



for sat in test['sat_id'].unique():

    train_X = train[train['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    test_X = test[test['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    target = train[train['sat_id']==sat][['x']]

    

    poly = PolynomialFeatures(degree= 2)

    train_poly_features = poly.fit_transform(train_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    test_poly_features = poly.fit_transform(test_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    

    poly_regression = LinearRegression()

    poly_regression.fit(train_poly_features,target)

    

    test_pred = poly_regression.predict(test_poly_features)

    pred = pd.DataFrame()

    pred['sat_id'] = [sat for i in range(test_X.shape[0])]

    pred['x'] = test_pred

    pred_x = pred_x.append(pred)

    pred_x = pred_x.reset_index(drop=True)
%%time

pred_y = pd.DataFrame()



for sat in test['sat_id'].unique():

    train_X = train[train['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    test_X = test[test['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    target = train[train['sat_id']==sat][['y']]

    

    poly = PolynomialFeatures(degree= 2)

    train_poly_features = poly.fit_transform(train_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    test_poly_features = poly.fit_transform(test_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    

    poly_regression = LinearRegression()

    poly_regression.fit(train_poly_features,target)

    

    test_pred = poly_regression.predict(test_poly_features)

    pred = pd.DataFrame()

    pred['sat_id'] = [sat for i in range(test_X.shape[0])]

    pred['y'] = test_pred

    pred_y = pred_y.append(pred)

    pred_y = pred_y.reset_index(drop=True)
%%time

pred_z = pd.DataFrame()



for sat in test['sat_id'].unique():

    train_X = train[train['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    test_X = test[test['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    target = train[train['sat_id']==sat][['z']]

    

    poly = PolynomialFeatures(degree= 2)

    train_poly_features = poly.fit_transform(train_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    test_poly_features = poly.fit_transform(test_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    

    poly_regression = LinearRegression()

    poly_regression.fit(train_poly_features,target)

    

    test_pred = poly_regression.predict(test_poly_features)

    pred = pd.DataFrame()

    pred['sat_id'] = [sat for i in range(test_X.shape[0])]

    pred['z'] = test_pred

    pred_z = pred_z.append(pred)

    pred_z = pred_z.reset_index(drop=True)
%%time

pred_Vx = pd.DataFrame()



for sat in test['sat_id'].unique():

    train_X = train[train['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    test_X = test[test['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    target = train[train['sat_id']==sat][['Vx']]

    

    poly = PolynomialFeatures(degree= 2)

    train_poly_features = poly.fit_transform(train_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    test_poly_features = poly.fit_transform(test_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    

    poly_regression = LinearRegression()

    poly_regression.fit(train_poly_features,target)

    

    test_pred = poly_regression.predict(test_poly_features)

    pred = pd.DataFrame()

    pred['sat_id'] = [sat for i in range(test_X.shape[0])]

    pred['Vx'] = test_pred

    pred_Vx = pred_Vx.append(pred)

    pred_Vx = pred_Vx.reset_index(drop=True)
%%time

pred_Vy = pd.DataFrame()



for sat in test['sat_id'].unique():

    train_X = train[train['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    test_X = test[test['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    target = train[train['sat_id']==sat][['Vy']]

    

    poly = PolynomialFeatures(degree= 2)

    train_poly_features = poly.fit_transform(train_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    test_poly_features = poly.fit_transform(test_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    

    poly_regression = LinearRegression()

    poly_regression.fit(train_poly_features,target)

    

    test_pred = poly_regression.predict(test_poly_features)

    pred = pd.DataFrame()

    pred['sat_id'] = [sat for i in range(test_X.shape[0])]

    pred['Vy'] = test_pred

    pred_Vy = pred_Vy.append(pred)

    pred_Vy = pred_Vy.reset_index(drop=True)
%%time

pred_Vz = pd.DataFrame()



for sat in test['sat_id'].unique():

    train_X = train[train['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    test_X = test[test['sat_id']==sat][['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']]

    target = train[train['sat_id']==sat][['Vz']]

    

    poly = PolynomialFeatures(degree= 2)

    train_poly_features = poly.fit_transform(train_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    test_poly_features = poly.fit_transform(test_X[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']])

    

    poly_regression = LinearRegression()

    poly_regression.fit(train_poly_features,target)

    

    test_pred = poly_regression.predict(test_poly_features)

    pred = pd.DataFrame()

    pred['sat_id'] = [sat for i in range(test_X.shape[0])]

    pred['Vz'] = test_pred

    pred_Vz = pred_Vz.append(pred)

    pred_Vz = pred_Vz.reset_index(drop=True)
sample['x'] =pred_x['x']

sample['y'] =pred_y['y']

sample['z'] =pred_z['z']

sample['Vx'] =pred_Vx['Vx']

sample['Vy'] =pred_Vy['Vy']

sample['Vz'] =pred_Vz['Vz']
sample.to_csv('poly_fit_deg_2.csv', index=False)
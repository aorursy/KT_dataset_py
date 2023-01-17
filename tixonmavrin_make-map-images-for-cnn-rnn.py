import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
train.sample(5)
test.sample(5)
train_lat_max = train['Lat'].max()

train_lat_min = train['Lat'].min()



train_long_max = train['Long'].max()

train_long_min = train['Long'].min()



train['Lat'] = (train['Lat'] - train_lat_min) / (train_lat_max - train_lat_min)

test['Lat'] = (test['Lat'] - train_lat_min) / (train_lat_max - train_lat_min)



train['Long'] = (train['Long'] - train_long_min) / (train_long_max - train_long_min)

test['Long'] = (test['Long'] - train_long_min) / (train_long_max - train_long_min)
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])
train = train.sort_values(by=['Date'])

test = test.sort_values(by=['Date'])
train['ConfirmedCases'].value_counts()
train['Fatalities'].value_counts()
CC_and_F_with_dates = []

for date in train['Date'].unique():

    train_date = train[train['Date'] == date]

    CC_and_F = train_date[['Lat', 'Long', 'ConfirmedCases', 'Fatalities']].values

    CC_and_F_with_dates.append(CC_and_F)
from scipy.interpolate import griddata



def gen_images(locs, features, n_gridpoints):

    feat_array_temp = []

    nElectrodes = locs.shape[0]

    

    assert features.shape[0] % nElectrodes == 0

    

    n_colors = features.shape[0] // nElectrodes

    

    for c in range(n_colors):

        feat_array_temp.append(features[c * nElectrodes : nElectrodes * (c+1)])



    grid_x, grid_y = np.mgrid[

                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,

                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j

                     ]

    

    temp_interp = []

    for c in range(n_colors):

        temp_interp.append(np.zeros([1, n_gridpoints, n_gridpoints]))



    for c in range(n_colors):

        temp_interp[c] = [griddata(locs, feat_array_temp[c], (grid_x, grid_y), method='cubic', fill_value=np.nan)]



    for c in range(n_colors):

        temp_interp[c] = np.nan_to_num(temp_interp[c])

        

    return np.swapaxes(np.asarray(temp_interp), 0, 1) # swap axes to have [samples, colors, W, H]

TRAIN = []

TEST = []



for i in range(len(CC_and_F_with_dates)-1):

    train_day = []

    test_day = []

    

    train_for_map = CC_and_F_with_dates[i]

    test_for_map = CC_and_F_with_dates[i+1]

    train_for_map = np.nan_to_num(train_for_map)

    

    generated_images = gen_images(train_for_map[:, :2], np.append(train_for_map[:, 2].T, train_for_map[:, 3].T), 64)

    images_train = [generated_images[0][0], generated_images[0][1]] # for ConfirmedCases and Fatalities

    

    for k, test_for_map_one in enumerate(test_for_map):

        

        test_for_map_zeros = np.zeros((test_for_map.shape[0],))

        test_for_map_zeros[k] = 1.0

        

        generated_images = gen_images(np.nan_to_num(test_for_map[:, :2]), test_for_map_zeros.T, 64)

        

        image_loc_state = generated_images[0][0] # location

        y = test_for_map_one[-2:] # test

        

        train_day.append(np.array(images_train + [image_loc_state]))

        test_day.append(y)

        

    train_day = np.array(train_day)

    test_day = np.array(test_day)



    TRAIN.append(train_day)

    TEST.append(test_day)

        

TRAIN = np.array(TRAIN)

TEST = np.array(TEST)
TRAIN.shape, TEST.shape
for i in range(58):

    plt.imshow(TRAIN[i, 120, 0])

    plt.show()
for i in range(58):

    plt.imshow(TRAIN[i, 120, 1])

    plt.show()
for i in range(58):

    plt.imshow(TRAIN[i, 120, 2])

    plt.show()
for i in range(50):

    plt.imshow(TRAIN[50, i, 2])

    plt.show()
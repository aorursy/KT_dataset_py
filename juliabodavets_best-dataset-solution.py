# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats # statistics

from matplotlib import pyplot as plt # visualize

from sklearn.metrics import mean_squared_error # metric



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train
# First 1000 features is a Poison distribution with different parameters

train.iloc[1,:1000].hist();
# Gaussian distribution is added( in modified features)

plt.hist(np.array(train.iloc[1,1000:2000]) - np.array(train.iloc[1,:1000]));
distrib_of_differ = np.array(train.iloc[:,1000:2000]) - np.array(train.iloc[:,:1000])

train_prediction = distrib_of_differ.mean(axis=1)
# Filter not Gaussian distributions in differs

board_p_value = 1e-3

normal = np.array([stats.normaltest((el - el.mean())/el.std()).pvalue > board_p_value for el in distrib_of_differ ])

normal
# Get standart diviations of Gaussian distributions

print(distrib_of_differ[train.target != 0].std(axis=1))

epselon = 0.5

good_std = np.array([1-epselon < el.std() < 1+epselon for el in distrib_of_differ ])

good_std
# Make all bad distribution predictions to zero

train_prediction[~(good_std & normal)] = 0
print('Metrics MSE: {}'.format(mean_squared_error(y_true=train.target, y_pred=train_prediction)))
def predcit_on_test(df):

    differ = np.array(df.iloc[:,1000:2000]) - np.array(df.iloc[:,:1000])

    pred = differ.mean(axis=1)

    

    first_b_m = np.array([stats.normaltest((el - el.mean())/el.std()).pvalue > board_p_value for el in differ ])

    second_b_m = np.array([1-epselon < el.std() < 1+epselon for el in differ ])

    

    pred[~(first_b_m & second_b_m)] = 0

    

    return pred



pred_test = predcit_on_test(pd.read_csv('../input/test.csv'))

pred_test
np.save('prediction.npy',pred_test)
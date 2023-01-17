import numpy as np

from scipy.stats import mode

import pandas as pd

from matplotlib.pyplot import hist

from sklearn.utils.extmath import weighted_mode



train_path = '../input/train.csv'

def convert(vec_str):

    days = np.array(vec_str.strip().split(' '), dtype = np.int)

    return days

def convert_to_dow(vec_str):

    tmp = np.array(vec_str.strip().split(' '), dtype = np.int)

    return np.apply_along_axis(lambda e: (e - 1) % 7 + 1, 0, tmp)



def mode_value(arr):

    dows = list(arr)

    return np.array([dows.count(i) for i in range(1,8)])



def weighted_mode_value(mode_weights, arr):

    local_mode_weighted = mode_weights[:len(arr)]

    val = weighted_mode(arr, local_mode_weighted)

    return int(val[0][0])

train = pd.read_csv(train_path, converters = {'id' : int, 'visits' : convert_to_dow})
mode_weights = np.arange(1.0,31.0,0.1)

dows = train.get('visits')

Y = np.array(list(map(lambda v: weighted_mode_value(mode_weights,v), dows)))

#hist(Y,20)

ids, answers = train.get('id'), Y
final_df = pd.DataFrame({'id' : ids, 'nextvisit' : answers})

final_df.to_csv('./weighted_mode.csv', index=False)
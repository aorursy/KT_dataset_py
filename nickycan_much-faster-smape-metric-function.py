# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

from numba import jit

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


class tick_tock:

    def __init__(self, process_name, verbose=1):

        self.process_name = process_name

        self.verbose = verbose

    def __enter__(self):

        if self.verbose:

            print(self.process_name + " begin ......")

            self.begin_time = time.time()

    def __exit__(self, type, value, traceback):

        if self.verbose:

            end_time = time.time()

            print(self.process_name + " end ......")

            print('time lapsing {0} s \n'.format(end_time - self.begin_time))

            

def smape_kun(y_true, y_pred):

    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))).fillna(0))



@jit

def smape_cpcm(y_true, y_pred):

    out = 0

    for i in range(y_true.shape[0]):

        a = y_true[i]

        b = y_pred[i]

        c = math.fabs(a)+math.fabs(b)

        if c == 0:

            continue

        out += math.fabs(a - b) / c

    out *= (200.0 / y_true.shape[0])

    return out
train = pd.read_csv("../input/train_1.csv")
y_true = train.iloc[:,-1].fillna(0)

y_pred = train.iloc[:,-2].fillna(0)                 
with tick_tock("time it"):

    print(smape_kun(y_true, y_pred))
with tick_tock("time it"):

    print(smape_cpcm(y_true, y_pred))
3.9470884799957275 / 0.04417014122009277
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from scipy import stats, integrate

import matplotlib.pyplot as plot

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

df_test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

df_macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])



dates = df_train['timestamp']

price_doc = df_train['price_doc']



plot.plot(dates, price_doc)
x, y = np.histogram(price_doc, bins=10)
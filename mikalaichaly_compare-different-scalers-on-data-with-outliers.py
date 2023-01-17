# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
dir = '../input'
print(os.listdir(dir))
# Any results you write to the current directory are saved as output.
from __future__ import print_function

import matplotlib as mpl
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

import seaborn as sns
sns.set(style="white", color_codes=True)

%matplotlib inline


import warnings
warnings.simplefilter('ignore')
dataset = pd.read_csv(dir + '/housing.csv')
dfX = pd.DataFrame(dataset['median_income'])
col = dfX['median_income'].values.reshape(-1, 1)

scalers = [
    #('Unscaled data', X),
    ('standard scaling', StandardScaler()),
    ('min-max scaling', MinMaxScaler()),
    ('max-abs scaling', MaxAbsScaler()),
    ('robust scaling', RobustScaler(quantile_range=(25, 75))),
    ('quantile transformation (uniform pdf)', QuantileTransformer(output_distribution='uniform')),
    ('quantile transformation (gaussian pdf)', QuantileTransformer(output_distribution='normal')),
    ('sample-wise L2 normalizing', Normalizer())
]

for scaler in scalers:
    dfX[scaler[0]] = scaler[1].fit_transform(col)
    
dfX.head()
orig = dfX['median_income']
orig_mean = orig.mean()
bins = 50
alpha=0.5

def plot_experiment(name):
    normalized = dfX[name]
    plt.figure(figsize=(10,5))
    plt.hist(orig, bins, alpha=alpha, label='Original')
    plt.axvline(orig_mean, color='k', linestyle='dashed', linewidth=1)

    plt.hist(normalized, bins, alpha=alpha, label=name)
    plt.axvline(normalized.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.legend(loc='upper right')

    plt.figure(figsize=(5,5))
    g = sns.jointplot(x="median_income", y=name, data=dfX, kind='hex', ratio=3)
    #sns.violinplot(x='median_income', data=dfX, )
    #sns.violinplot(x='standard scaling', data=dfX)
    #plt.boxplot(dfX['median_income'])
    #plt.boxplot(dfX['standard scaling'])
    plt.show()

plot_experiment('standard scaling')
plot_experiment('min-max scaling')
plot_experiment('max-abs scaling')
plot_experiment('robust scaling')
plot_experiment('quantile transformation (uniform pdf)')
plot_experiment('quantile transformation (gaussian pdf)')
plot_experiment('sample-wise L2 normalizing')
dfX[['median_income', 'sample-wise L2 normalizing']].sample(20)
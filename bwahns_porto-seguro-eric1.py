# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import pandas as pd

#sample_submission = pd.read_csv("../input/porto-seguro-safe-driver-prediction/sample_submission.csv")

#test = pd.read_csv("../input/porto-seguro-safe-driver-prediction/test.csv")

#train = pd.read_csv("../input/porto-seguro-safe-driver-prediction/train.csv")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel

from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier



pd.set_option('display.max_columns', 100)
train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')
train.head()
# check tail

train.tail()
train.shape
train.drop_duplicates()

train.shape
test.shape
test.head()

# 정말 target column이 없다.

# 주어진 데이타로 complain 하는 건 인지 아닌 건 인지 알고 싶으니.
train.info()
print(train.target.unique())
print(sum(train['target'] == 1) / 595212 * 100)
test['target']
test['target'] = np.nan
test['target']
# column과 data, hue를 받아서, hue값에 따른 seaborn countplot

def bar_plot(col, data, hue=None):

    f, ax = plt.subplots(figsize=(10,5))

    sns.countplot(x=col, hue=hue, data=data, alpha=0.5)

    plt.show()



# histogram, data[col] 사용해서, 해당 컬럼 histogram 

def dist_plot(col, data):

    f, ax = plt.subplots(figsize=(10,5))

    sns.distplot(data[col].dropna(), kde=False, bins=10)

    plt.show()



# target 기준으로 column 값과 target 관계 살펴봄

def bar_plot_ci(col, data):

    f, ax = plt.subplots(figsize=(10,5))

    sns.barplot(x = col, y='target', data=data)

    plt.show()
#이진 변수 list

binary = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',

          'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin',

          'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']



# 범주형 변수

category = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',

            'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',

            'ps_car_10_cat', 'ps_car_11_cat']



# 정수형 변수

integer = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',

           'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13',

           'ps_calc_14', 'ps_car_11']



# 소수형 변수

floats = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_car_12', 'ps_car_13',

          'ps_car_14', 'ps_car_15']
for col in binary:

    bar_plot(col, train)
for col in category:

    bar_plot(col, train)
for col in integer:

    bar_plot(col, train)
for col in floats:

    bar_plot(col, train)
corr_data = train.corr()

f, ax = plt.subplots(figsize=(15,7))

sns.heatmap(corr_data, cmap='summer')
for col in binary:

    bar_plot_ci(col, train)
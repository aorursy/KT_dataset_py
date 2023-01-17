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
#1918 Pandemic Influenza Mortality, Chicago USA

#Source https://figshare.com/articles/1918_Pandemic_Influenza_Mortality_Chicago_USA/9172790/1

import pandas as pd

pandemic1918 = pd.read_csv("../input/points.csv")
pandemic1918.head()
pandemic1918.dtypes
pandemic1918.shape
def display_all(df):

    with pd.option_context("display.max_rows", 1000):

        with pd.option_context("display.max_columns", 1000):

            display(df)



print(display_all(pandemic1918.describe(include='all').transpose()))
print(pandemic1918.isnull().sum())
#インフルエンザ死亡者数、インフル＋肺炎死亡者数割合

import matplotlib.pyplot as plt

label_pneumonia = ['influenza death', 'influenza and pneumonia death']

plt.pie(pandemic1918['pneumonia'].value_counts(), labels=label_pneumonia)

plt.show()
#週ごとの累積死亡者数

plt.hist(pandemic1918['week'].value_counts(), cumulative=True)
#パンデミック発生場所

pandemic1918_avxy_0 = pandemic1918[(pandemic1918['x (m)'] > 1) | (pandemic1918['y (m)'] > 1) & (pandemic1918['pneumonia'] == 0)]

pandemic1918_avxy_1 = pandemic1918[(pandemic1918['x (m)'] > 1) | (pandemic1918['y (m)'] > 1) & (pandemic1918['pneumonia'] == 1)]

plt.scatter(x=pandemic1918_avxy_0['x (m)'], y=pandemic1918_avxy_0['y (m)'], label=["influenza death"], alpha=1.0, s=5)

plt.scatter(x=pandemic1918_avxy_1['x (m)'], y=pandemic1918_avxy_1['y (m)'], label=["influenza and pneumonia death"], alpha=0.3, s=5)

plt.legend()

plt.show()
#週毎発生位置

for i in range(7):

    pandemic1918_avxy_week = pandemic1918[(pandemic1918['x (m)'] > 1) | (pandemic1918['y (m)'] > 1) & (pandemic1918['week'] == i)]

    plt.scatter(x=pandemic1918_avxy_week['x (m)'], y=pandemic1918_avxy_week['y (m)'], c=pandemic1918_avxy_week['week'].astype('float64'), cmap='Reds_r', vmin=1, vmax=7, alpha=0.3, s=5)



plt.colorbar()

plt.show()
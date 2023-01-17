# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load data

youtube=pd.read_csv('/kaggle/input/youtube-new/INvideos.csv')
#correlation between views and likes

youtube.corr()['likes'].abs()
#draw heat map

fig, ax = plt.subplots(figsize=(11,9))

sns.heatmap(youtube.corr(), cmap='RdBu', vmax=.3, center=0, square=True, linewidths=0.5)
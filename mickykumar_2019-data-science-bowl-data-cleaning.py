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

# Any results you write to the current directory are saved as output.

sub = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

sub.to_csv('submission.csv',index = 'false')
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
train.isnull()
test.isnull()
import seaborn as sns

import matplotlib.pyplot as plt

def chk_corr(df):

    corrs = train.corr()

    plt.figure(figsize = (7,7))

    # Heatmap of correlations

    sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

    plt.title('Correlation Heatmap');
chk_corr(train)
def chk_corr(df):

    corrs = test.corr()

    plt.figure(figsize = (7,7))

    # Heatmap of correlations

    sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

    plt.title('Correlation Heatmap');
chk_corr(test)
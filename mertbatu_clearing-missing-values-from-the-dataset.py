# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib_venn as venn

from math import pi

from pandas.tools.plotting import parallel_coordinates

import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/aps_failure_test_set.csv")
data.info()
data.head()
sample_data = data.loc[:,['ab_000','ac_000','ad_000']]

sample_data.ab_000.replace(['na'],np.nan, inplace = True)

sample_data.ac_000.replace(['na'],np.nan, inplace = True)

sample_data.ad_000.replace(['na'],np.nan, inplace = True)

data_missingno = pd.DataFrame( sample_data.head(20))
data_missingno
import missingno as msno

msno.matrix(data_missingno)

plt.show()
msno.bar(data_missingno)

plt.show()
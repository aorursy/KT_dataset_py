# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.graph_objs as go

import plotly.tools as tls

from plotly.offline import iplot, init_notebook_mode

#import cufflinks

#import cufflinks as cf

import plotly.figure_factory as ff



from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression



from functools import partial

from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')

test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')
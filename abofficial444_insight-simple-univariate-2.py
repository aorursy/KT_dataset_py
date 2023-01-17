## import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

import seaborn as sns

#Ignore annoying warning from sklearn and seaborn

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn



#other libraiaries

import os

import copy

from collections import defaultdict

from collections import Counter

from sklearn import metrics

import matplotlib.pyplot as plt

%matplotlib inline

import os

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

import re

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/training_v2.csv');train.head()
train.icu_type.value_counts().to_frame()
h = train[(train['gender'] == 'M') & (train['icu_type'] == 'Med-Surg ICU')]

h.sample(10)
plt.figure(figsize=[10,8])

sns.countplot(x='icu_type', hue='hospital_death', data=train)
plt.figure(figsize=[10,8])

sns.countplot(x='icu_type', hue='gender', data=train)
train.readmission_status.value_counts()
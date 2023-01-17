# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing required packages

import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import seaborn as sns
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

from scipy import stats
from matplotlib import rc

from pylab import rcParams
rcParams['figure.figsize'] = 15,5

from sklearn import metrics
from tensorflow import keras

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'
register_matplotlib_converters()

from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D

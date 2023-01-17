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
from sklearn.preprocessing import LabelEncoder

from plotly.offline import init_notebook_mode

from datetime import datetime, timedelta

init_notebook_mode(connected=False)

from keras.models import Sequential

import plotly.graph_objects as go

import matplotlib.pyplot as plt

from fbprophet import Prophet

import plotly.express as px

import plotly.offline as py

from keras import layers

import seaborn as sns

import pandas as pd

import numpy as np

import gc

import os
print(os.listdir("../input/novel-corona-virus-2019-dataset/"))

path="../input/novel-corona-virus-2019-dataset/"
df_one = pd.read_csv(path+'time_series_covid_19_recovered.csv')
df_one.head()
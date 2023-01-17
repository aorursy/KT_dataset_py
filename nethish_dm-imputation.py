# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

import plotly.express as px

import plotly

from plotly.subplots import make_subplots

import plotly.graph_objects as go
indices = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'Married-spouse-absent', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

mel_data = pd.read_csv('/kaggle/input/melbourne/melb_data.csv')

adult_data = pd.read_csv('/kaggle/input/adults/adult.csv', names=indices)



workclass = adult_data.workclass

area = mel_data.BuildingArea

NAN = float('nan')

max_area = [1368, 1484, 1588, 2234, 2560, 2830, 3640, 13245]

area = area.drop(max_area)



workclass_cats = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']

# fig = px.box(area)

# fig.show()



fig = make_subplots(rows=1, cols=2)

fig.append_trace(go.Box(x=area, text=""), row=1, col=1)

fig.append_trace(go.Histogram(x=area, text=""), row=1, col=2)



fig.update_layout(height=600, width=750, title_text =  "Before imputation")

fig.show()



area.describe()

# workclass.value_counts().plot(kind='bar', figsize=(50, 10))

workclass.value_counts().plot(kind='pie', figsize=(10, 10))

workclass.describe()
def impute_mean(_data):

    data = _data

    dropped = _data.dropna()

    mean = np.mean(data)

    data = data.fillna(mean)

    dropped = _data.dropna()

    

    fig = make_subplots(rows=1, cols=2)

    fig.append_trace(go.Box(x=data, text=""), row=1, col=1)

    fig.append_trace(go.Histogram(x=data, text=""), row=1, col=2)



    fig.update_layout(height=600, width=750, title_text =  "Mean Imputation")

    fig.show()

    return data



impute_mean(area).describe()
def impute_median(_data):

    data = _data

    dropped = _data.dropna()

    median = np.median(dropped)

    data = data.fillna(median)

    fig = make_subplots(rows=1, cols=2)

    fig.append_trace(go.Box(x=data, text=""), row=1, col=1)

    fig.append_trace(go.Histogram(x=data, text=""), row=1, col=2)



    fig.update_layout(height=600, width=750, title_text =  "Median Imputation")

    fig.show()

#     fig = px.box(data, points='all')

#     fig.show()

    return data



impute_median(area).describe()    

def impute_random(_data):

    data = np.array(_data)

    dropped = _data.dropna()

    data = data.reshape(1, -1)

    my_imputer = SimpleImputer()

    data = (my_imputer.fit_transform(data))[0]

    fig = make_subplots(rows=1, cols=2)

    fig.append_trace(go.Box(x=data, text=""), row=1, col=1)

    fig.append_trace(go.Histogram(x=data, text=""), row=1, col=2)



    fig.update_layout(height=600, width=750, title_text =  "Random Imputation")

    fig.show()

#     fig = px.box(data, points='all')

#     fig.show()

    return pd.Series(data)



impute_random(area).describe()
def impute_mode(_data):

    data = _data

    mode = data.mode()[0]

    data = data.replace(' ?', mode)

    fig, axs = plt.subplots(1, 2, figsize=(30, 8))

    data.value_counts().plot(kind='pie', ax=axs[0])

    data.value_counts().plot(kind='bar', ax=axs[1])

    fig.show()

    return data



impute_mode(workclass).describe()
def impute_missing(_data):

    data = _data

    data = data.replace(' ?', 'Missing')

    fig, axs = plt.subplots(1, 2, figsize=(30, 8))

    data.value_counts().plot(kind='pie', ax=axs[0])

    data.value_counts().plot(kind='bar', ax=axs[1])

    fig.show()

    return data



impute_missing(workclass).describe()
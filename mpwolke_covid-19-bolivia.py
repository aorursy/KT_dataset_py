# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cronologia-decesos-covid19-bolivia/cronologia_covid19_bolivia_decesos.csv', encoding='ISO-8859-2')

df.head()
plt.style.use('dark_background')

plt.figure(figsize=(8,6))

sns.countplot(df["edad"])

plt.xticks(rotation=90)

plt.show()
plt.style.use('dark_background')

sns.distplot(df.nro)
df.dtypes
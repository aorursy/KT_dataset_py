# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ftrain = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

print("import load train")

ftest = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

print("import load test")

fsubs = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

print("import complete")

print("Edit Test Complete")
print("train info")

ftrain.info()

print("test info")

ftest.info()

print("subs info")

fsubs.info()
ftrain.shape, ftest.shape, fsubs.shape

ftrain.head()
ftrain.drop(['Province_State'], axis=1, inplace=True)

ftest.drop(['Province_State'], axis=1, inplace=True)
ftrain.head()

ftrain.describe()
ftrain.info()
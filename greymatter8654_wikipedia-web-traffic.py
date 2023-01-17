# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import math

import seaborn as sns

import calendar



%matplotlib inline

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from bokeh.charts import TimeSeries, show
for f in os.listdir('../input'):

    size_bytes = round(os.path.getsize('../input/' + f)/ 1000, 2)

    size_name = ["KB", "MB"]

    i = int(math.floor(math.log(size_bytes, 1024)))

    p = math.pow(1024, i)

    s = round(size_bytes / p, 2)

    print(f.ljust(25) + str(s).ljust(7) + size_name[i])
train_df = pd.read_csv("../input/train_1.csv")

key_df = pd.read_csv("../input/key_1.csv")
print("Train".ljust(15), train_df.shape)

print("Key".ljust(15), key_df.shape)
print(train_df[:4].append(train_df[-4:], ignore_index=True))
print(key_df[:4].append(key_df[-4:], ignore_index=True))
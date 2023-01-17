# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)
data = pd.read_csv("../input/911.csv")

list(data)

data.head()
data["title"].value_counts()
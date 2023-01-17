# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import locale



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/zomato.csv")
print(data.columns)
feat = ['name', 'online_order','rate','location','rest_type','approx_cost(for two people)']

ftData = data[feat]
ftData.dropna(how='any')
cost = [x[5] for x in ftData.values if x[3] == 'Banashankari']

rate = [x[2] for x in ftData.values if x[3] == 'Banashankari']

#float("50".replace(',',''))
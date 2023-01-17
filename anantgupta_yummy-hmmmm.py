# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/menu.csv")



# ANALYSIS 1

# My aim is to find some items which cannot go hand in hand

colValues=list(data.columns.values)

colValues.remove('Category')

colValues.remove('Item')

colValues.remove('Serving Size')



corr=data[colValues].corr()

sns.heatmap(corr)



# We can see from the following chart that SUGAR is highly uncorrelated with 

# a) Sodium

# b) Dietary Fiber

# c) Iron
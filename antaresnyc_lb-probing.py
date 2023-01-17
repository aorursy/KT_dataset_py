# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()

classes = {'Cover_type1' : 0.37053,'Cover_type2' : 0.49657,'Cover_type3' : 0.059647,'Cover_type4' : 0.00106,'Cover_type5' : 0.01287,'Cover_type6' : 0.02698,'Cover_type7' : 0.03238}

pd_cl = pd.DataFrame.from_dict(classes, orient='index', columns=['Share'])
fig, ax = plt.subplots(figsize=(10,10))

sns.barplot(x = pd_cl.index, y = pd_cl.Share  *100,  data = pd_cl, ax = ax)
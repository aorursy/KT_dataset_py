# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #imported for visualization





df=pd.read_csv("../input/cereal.csv", low_memory=False)

df_fat=df["fat"]



plt.hist(df_fat, label="Fat in ceral products")

#dont get why it does not give the label to the printed histogram

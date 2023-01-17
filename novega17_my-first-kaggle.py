# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# view the data

df_16=pd.read_csv('../input/2016.csv')

df_16.head()

# learn details about the dataset

df_16.describe()

# pratice data visualization grouping by region

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

topREG = df_16.sort_values('Happiness Score', ascending=False).head(15)

topREG = sns.barplot(data = df_16, x = "Happiness Score", y = "Region",)

topREG.axes.set_title("Happiness Rank of World Regions",fontsize=30)

topREG.set_xlabel("Happiness Score",fontsize=30)

topREG.set_ylabel("Region Name",fontsize =20)

topREG.tick_params(labelsize = 25)



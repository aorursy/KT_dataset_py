# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline



# Any results you write to the current directory are saved as output.
df15=pd.read_csv('../input/2015.csv',encoding = "ISO-8859-1")

df16=pd.read_csv('../input/2016.csv',encoding = "ISO-8859-1")

df15.dropna(inplace=True)

df15=df15[df15['age']!='Unknown']

df15.age=df15.age.astype(np.int)
df15.head()
df15.shape
race = df15["raceethnicity"].unique()
df15.groupby("raceethnicity")


sns.stripplot(x='raceethnicity',y='age',hue='gender',data=df15)
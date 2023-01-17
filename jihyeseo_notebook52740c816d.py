# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

filename = check_output(["ls", "../input"]).decode("utf8").strip()

df = pd.read_csv("../input/" + filename, thousands=",")

print(df.dtypes)

df.head()
df['JJJ Start']= pd.to_datetime(df['JJJ Start'])

df['JJJ Peak']= pd.to_datetime(df['JJJ Peak'])

df['JJJ End']= pd.to_datetime(df['JJJ End'])
df.dtypes
df.nunique()
df['JJJ Class'].value_counts()
df['Class'] = df['JJJ Class'].str[0]
df['Class'].value_counts()
df.groupby('Class').mean()
df['duration'] = df['JJJ End'] - df['JJJ Start']
df['duration'].mean()
df.groupby('Class')['duration'].mean()
df.groupby('Class')['duration'].describe()
df.groupby('JJJ Class')['duration'].describe()
df['peakFast'] = df['JJJ Peak'] - df['JJJ Start']
df['relativePeak'] = df['peakFast']/df['duration']
df.groupby('Class')['peakFast'].describe()
df.groupby('Class')['relativePeak'].describe()
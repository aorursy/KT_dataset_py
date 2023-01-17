# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
filename = check_output(["ls", "../input"]).decode("utf8").strip()

print(filename)
df = pd.read_csv("../input/" + filename, engine = 'python', sep = '\t' )
df.columns = ['howto']
df.head()
how2clean = df[df['howto'].str.contains('clean')]

how2clean
how2clean = df[df['howto'].str.contains('make')]

how2clean
how2clean = df[df['howto'].str.contains('cook')]

how2clean
wordCounts = df.howto.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
popularWords = wordCounts.sort_values(ascending = False)/len(df)

popularWords
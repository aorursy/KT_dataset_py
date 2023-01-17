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
filenames = check_output(["ls", "../input"]).decode("utf8").split('\n')

df = pd.read_csv("../input/" + filenames[0], thousands=",", engine = 'python', skipfooter = 40)

df['YEAR'] = df['YEAR'].str.rstrip("*")



df = df.dropna(axis = 0, how = 'all')

df['YEAR'] = df['YEAR'].astype('int32')
df.dtypes
df.head()
df.tail()
df.plot(x = 'YEAR', y = 'POUNDS GRADED')
df.plot(x = 'YEAR', y = 'PRIME')
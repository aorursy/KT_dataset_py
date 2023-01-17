# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('hello world')
ds = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")

ds.info()
["main_category","state"]

ds[["main_category","state"]]
ds.plot()
ds["state"].value_counts()
projState = ds["state"].value_counts()

projState.plot(kind = 'pie')
projState.plot(kind = 'bar')
projState.plot(kind = 'barh')
ds.plot(x = 'deadline', y = 'goal', kind = 'line')
ds.head(100).plot(x = 'deadline', y = 'goal', kind = 'line')
ds.plot(x = 'deadline', y = 'goal', kind = 'scatter')
ds.hist(column = 'goal')
under5000 = ds[(ds["goal"] < 5000)]

under5000.hist(column='goal')
under5000.hist(column = 'goal', bins = 20)
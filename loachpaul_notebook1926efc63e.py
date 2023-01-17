# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from datetime import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/ShanghaiPM20100101_20151231_Training - Training.csv')

df_test = pd.read_csv("../input/ShanghaiPM20100101_20151231_Test - Test.csv")
df.head(3)
df_test.head(3)
year_2013 = df[df['year'] == 2013]

year_2014 = df[df['year'] == 2014]

year_2015 = df[df['year'] == 2015]
year_2013["PM_US Post"].mean()
year_2014["PM_US Post"].mean()
year_2015["PM_US Post"].mean()
df_dropped = df.dropna()

df.plot('year','PM_Jingan')
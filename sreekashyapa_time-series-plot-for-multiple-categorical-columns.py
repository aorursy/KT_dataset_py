# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import necessary packages
import pandas as pd
import datetime
from random import randint
%matplotlib inline
import matplotlib.pyplot as plt
#create sample ts data with two categorical columns
tst = pd.DataFrame(index=pd.date_range(datetime.datetime(2010, 1, 1), end=datetime.datetime(2010, 2, 1), freq='D'), data=[[randint(0, 1), randint(0, 1)] for x in range(32)])
tst.columns=["CallBack","Repair"]
tst.head()
#plot two categorical variables against time
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(tst.index, tst.CallBack)
plt.scatter(tst.index-1, tst.Repair,c='r')
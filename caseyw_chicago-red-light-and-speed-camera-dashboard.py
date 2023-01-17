# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/speed-camera-violations.csv")
data_grp=data[['CAMERA ID','VIOLATIONS']].groupby(by='CAMERA ID',as_index=False).sum()
data_top10=data_grp.sort_values(by='VIOLATIONS',ascending=False).head(10)
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = sns.barplot(y=data_top10['CAMERA ID'],x=data_top10['VIOLATIONS'])
plt.show()

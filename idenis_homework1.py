# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/USvideos.csv")
data.info()
data.head(10)
data.corr()
import seaborn as sns
a = sns.heatmap(data.corr())
data.columns
data.likes.plot(kind='line', color='blue', alpha=0.5, grid=True, label='Likes')
data.comment_count.plot(kind='line', color='red', alpha=0.5, grid=True, label='Likes')
data.views.plot(kind='line', color='green', alpha=0.5, grid=True, label='Likes')

data.plot(kind='scatter', x='likes', y='comment_count', color='red', alpha=0.5, grid=True)
plt.xlabel='likes'
plt.ylabel='views'
plt.title='A'
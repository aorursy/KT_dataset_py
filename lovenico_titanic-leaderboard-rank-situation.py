# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pyplot as plt
% matplotlib inline

# Any results you write to the current directory are saved as output.
filename = "../input/titanic_leaderboard.csv"
data = pd.read_csv(filename, error_bad_lines=False)
data.info()
print("_"*50)
print (data.head())
print("_"*50)
data.describe()
from datetime import datetime
import matplotlib.dates as mdates
#data = data.iloc[:24000, :]  #数据切片
plt.figure(figsize = (30,5))
dates = data.SubmissionDate
xs = [datetime.strptime(d, '%Y/%m/%d %H:%M').date() for d in dates]
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H:%M'))   #匹配时间格式
#plt.gca().xaxis.set_major_locator(mdates.DayLocator())                                             #时间条目分布

plt.scatter(xs, data.Score, color = "B",alpha=1, marker = '.')
plt.gcf().autofmt_xdate()  # 自动旋转日期标记
plt.title("Titanic_Rank_Situation ")
#plt.savefig("rankSituation.png")
plt.show()

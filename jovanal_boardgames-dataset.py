# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/20000-boardgames-dataset/boardgames1.csv')
df.head()
df.info()
voteUmur = df.groupby(['playerage'])['objectid'].count()
#voteUmur
plt.bar(voteUmur.index, voteUmur)
plt.ylabel("Jumlah Vote")
plt.xlabel("Age")
plt.title("Vote Result Player Age")
plt.show()
MinUmur = df.groupby(['minage'])['objectid'].count()
plt.barh(MinUmur.index, MinUmur)
plt.ylabel("Min Umur")
plt.xlabel("Jumlah Boardgame")
plt.title("Minimal Umur Player")
plt.show()
tahunPublish = df.groupby(['yearpublished'])['objectid'].count()
plt.bar(tahunPublish.index, tahunPublish)
plt.xlabel("Tahun Publish")
plt.ylabel("Jumlah Boardgame")
plt.xlim(1999, 2020)
plt.title("Tahun Publish Boardgame")
plt.show()
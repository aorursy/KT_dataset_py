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
df = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv')
df.head()
df.info()
perbandingan = df.groupby(['international_students'])['world_rank'].count()
perbandingan

plt.barh(perbandingan.index, perbandingan)
plt.xlabel('Peringkat Dunia')
plt.ylabel('Mahasiswa International')
plt.show()
penelitian = df.groupby(['research'])['world_rank'].count()
penelitian

plt.bar(penelitian.index, penelitian)
plt.xlabel('Penelitian')
plt.ylabel('Mahasiswa International')
plt.show()
perbandingan = df.groupby(['country'])['teaching'].count()
perbandingan

plt.barh(perbandingan.index, perbandingan)
plt.xlabel('Kota')
plt.ylabel('Pengajaran')
plt.show()
perbandingan = df.groupby(['country'])['international'].count()
perbandingan

plt.barh(perbandingan.index, perbandingan)
plt.xlabel('Kota')
plt.ylabel('International')
plt.show()
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
df = pd.read_csv("../input/Data-Jumlah-Penduduk-Berdasarkan-Pendidikan-Tahun-2019.csv")



print(df.head())
print(df.describe(include=[np.object]))
import seaborn as sns

import matplotlib.pyplot as plt
sns.set_style('whitegrid')



data = df.groupby(['nama_kabupaten', 'pendidikan'])['jumlah'].agg('sum').sort_values()



bar = sns.barplot(x=data.loc["JAKARTA BARAT"].index, y=data.loc["JAKARTA BARAT"], palette="Blues_d")



bar.set_xticklabels(bar.get_xticklabels(), rotation=45, horizontalalignment='right')



plt.title('Sebaran Jenjang Pendidikan Masyarkat JAKBAR')

plt.show()
kabupaten = df.nama_kabupaten.unique().tolist()



fig = plt.figure(figsize=(20,20))

axes = dict()

for i in range(1, len(kabupaten)+1):

    fig.add_subplot(int('23'+str(i)))

    plt.title(kabupaten[i-1], fontsize=14)

    axes[int('23'+str(i))] = sns.barplot(x=data.loc[kabupaten[i-1]].index, y=data.loc[kabupaten[i-1]], palette="Blues_d")

    axes[int('23'+str(i))].set_xticklabels(axes[int('23'+str(i))].get_xticklabels(), rotation=20, horizontalalignment='right')

    

plt.show()

    
index = np.arange(len(df.pendidikan.unique()))

bar_width = 0.35



fig, ax = plt.subplots()



jaksel = ax.bar(index, data.loc['JAKARTA SELATAN'], 

                bar_width, label='JAKARTA SELATAN')



jakpus = ax.bar(index+bar_width, data.loc['JAKARTA PUSAT'],

                 bar_width, label="JAKARTA PUSAT")



ax.set_xlabel('Pendidikan')

ax.set_ylabel('Jumlah')

ax.set_title('JAKARTA SELATAN VS JAKARTA PUSAT')

ax.set_xticks(index + bar_width / 2)

ax.set_xticklabels(data.loc["JAKARTA SELATAN"].index.tolist(), rotation=20, horizontalalignment='right')

ax.legend()



plt.show()
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
'/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df.info()
#pake mean & grouping, tipe : bar



gaji = df.groupby(['degree_t'])['salary'].mean()



plt.bar(gaji.index, gaji)

plt.xlabel("Jurusan")

plt.ylabel('Gaji')



plt.show()
#

sns.countplot(df['status'], hue=df['gender'])
sns.countplot(df['specialisation'], hue=df['degree_t'])
sns.countplot(y="degree_t", hue="hsc_s", data=df)

#barh + count

mhs_jurusan = df.groupby(['degree_t'])['gender'].count()



plt.barh(mhs_jurusan.index, mhs_jurusan)



plt.ylabel("Jurusan")

plt.xlabel('Jumlah Mahasiswa')



plt.show()
# pakai plot

df.plot(x = 'sl_no', y = 'etest_p')

fig, ax = plt.subplots()

df.groupby(['gender']).plot(x = 'sl_no', y = 'mba_p', ax=ax)



fig, ax = plt.subplots()



ax.scatter(df['etest_p'], df['salary'])



fig, ax = plt.subplots()



ax.scatter(df['etest_p'], df['salary'], color='coral')

ax.scatter(df['mba_p'], df['salary'], color='lightgreen')





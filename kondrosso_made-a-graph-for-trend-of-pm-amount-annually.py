# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/ShanghaiPM20100101_20151231_Training - Training.csv')
ax1 = fig.add_subplot(2,2,1)
df.describe()
df.groupby('year')[PM_Jingan].sum()
df.groupby('year')['PM_Jingan'].sum()
df.groupby('year')['PM_Jingan'].avg()
df_years
df.groupby('year')['PM_Jingan'].min()
df_list = list(df.groupby('year')['PM_Jingan'])
df_list
df_list[1]
df["year"].describe()
df_list
year_2015
df_2010 = [df['year'] == 2010]
df_2010.describe()
df_2010
df
df.dtype
df['PM_Jingan']
df_2010 = df['PM_Jinnan'[df['year'] == 2010]]
df_2010 = df['PM_Jingan']
df_2010
df_2010 = df['PM_Jingan'[df['year'] == 2010]]
df_2010 = df['PM_Jingan'[df['year'] == "2010"]]
df_2010 = [df['year'] == 2010]
df_2010
df_2010 = df['PM_Jingan'][df['year'] == 2010]
df_2010
df_2012 = df['PM_US Post'][df['year'] == 2012]
df_2012.describe()
df_2013 = df[['PM_Jingan', 'PM_US Post', 'PM_Xuhui']][df['year'] == 2013]
df_2013.describe()
df_describe2013 = df_2013.describe()
df_2014 = df[['PM_Jingan', 'PM_US Post', 'PM_Xuhui']][df['year'] == 2014]
df_2014.describe()
df_2015 = df[['PM_Jingan', 'PM_US Post', 'PM_Xuhui']][df['year'] == 2015]
df_mean = pd.concat([df_2013_describe[1:2], df_2014_describe[1:2], df_2015_describe[1:2]])
df_2015.describe()
df_describe2014
df_mean.index = ['2013', '2014', '2015']
df_mean
df_mean.columns = ['Jingan', 'US', 'Xuhui']
df_2015_describe = df_2015.describe()
df_2014_describe = df_2014.describe()
df_2013_describe = df_2013.describe()
df_2015_describe
df_2013_describe.columns = ['PM_Jingan2013', 'PM_US Post2013', 'PM_Xuhui2013']
df_2013_describe
df_2014_describe
df_2014_describe.columns = ['PM_Jingan2014', 'PM_US Post2014', 'PM_Xuhui2014']
df_mean
df_2014_describe
import matplotlib.pyplot as plt
plot(df_mean)
plt.xlabel('xlabel')
plot(df_mean)

plt.xlabel('year')

plt.ylabel('PM amount')

plt.title('PM trend')
fig = plt.figure()
ax3 = fig.add_subplot(2,2,3)
plt.plot(df_mean, 'c--')

plt.xlabel('year')

plt.ylabel('PM amount')

plt.title('PM trend')

plt.subplots(3,3)

ax1 = plt.plot(df_mean)

plt.xlim(2013, 2015)

plt.ylim(45, 65)

plt.sharex
plt.subplots(3,3)

ax1 = plt.plot(df_mean)

plt.xlim(2013, 2015)

plt.ylim(45, 65)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
plt.plot(df_mean, 'c-.')

plt.xlabel('year')

plt.ylabel('PM amount')

plt.title('PM trend')
ax=plt.plot(df_mean, 'k-.', label='Default')

plt.plot(df_mean, 'c-', drawstyle='steps-post', label='steps-post')

plt.xticks([2013, 2014, 2015], rotation = 30, fontsize='large')

plt.xlabel('year')

plt.ylabel('PM amount')

plt.title('PM trend')

plt.legend(loc='best')

ax.text(2, 16,'Hello world!', family = 'monospace', fontsize = 10)
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
open_data= '../input/open-interest-15052020/securities.csv'

data=pd.read_csv(open_data, ';', skiprows = 2)

data.head()

test_1=data.groupby(['ticker','clgroup'])['pos','pos_long','pos_short','pos_long_num','pos_short_num'].sum()

test_1

#test_filter=test_1.groupby(['ticker','clgroup','pos'])['pos_long','pos_short','pos_long_num','pos_short_num'].sum().sort_values(by='pos', ascending=False)

#test_filter.head(10)
tt=test_1.pos_long_num/test_1.pos_short_num

tt.sort_values(ascending=False).head(10)
t=test_1.pos_long+test_1.pos_short

short=t.sort_values().head(10)

short

#test_1.append(t)
u=t.sort_values().tail(10)

long=u.sort_values(ascending=False)

long


fiz=data.loc[data['clgroup'] == 'FIZ'].sort_values(by='pos', ascending=True)

fiz_data=fiz.groupby('ticker')['pos','pos_long','pos_short','pos_long_num','pos_short_num'].sum().sort_values(by='pos', ascending=True).head(10)

print('FIZ')

fiz_data
print('YUR')

yur=data.loc[data['clgroup'] == 'YUR'].sort_values(by='pos', ascending=True)

yur_data=yur.groupby('ticker')['pos','pos_long','pos_short','pos_long_num','pos_short_num'].sum().sort_values(by='pos', ascending=True).head(10)

yur_data
yur_data.pos
import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
sns.set_style("darkgrid")

plt.figure(figsize=(15,8))

sns.barplot(x=fiz_data.index,y=fiz_data['pos'])

sns.barplot(x=fiz_data.index,y=fiz_data['pos_long'])

sns.barplot(x=fiz_data.index,y=fiz_data['pos_short'])

plt.ylabel('position')

plt.title('FIZ Traders')
sns.set_style("darkgrid")

plt.figure(figsize=(15,8))

sns.barplot(x=yur_data.index,y=yur_data['pos'])

sns.barplot(x=yur_data.index,y=yur_data['pos_long'])

sns.barplot(x=yur_data.index,y=yur_data['pos_short'])

plt.ylabel('position')

plt.title('YUR Traders')
data1=pd.read_csv(open_data, ';', skiprows = 2, index_col='ticker', parse_dates=True)



t=data1.groupby(['ticker'])['pos','pos_long','pos_short','pos_long_num','pos_short_num'].sum()

sns.set_style('darkgrid')

plt.figure(figsize=(19,10))

sns.lineplot(data=t)
sns.set_style('darkgrid')

plt.figure(figsize=(19,10))

sns.barplot(x=t.index,y=t['pos'])

sns.barplot(x=t.index,y=t['pos_long'])

sns.barplot(x=t.index,y=t['pos_short'])

plt.ylabel('position')
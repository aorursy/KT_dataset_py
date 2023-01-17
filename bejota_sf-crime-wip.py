# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # viz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_df = pd .read_csv("../input/train.csv",parse_dates=['Dates'])
train_df.shape
train_df.columns
train_df.head(5)
train_df.describe(include='all')
train_df.Category.value_counts()
num = 20
train_df.Category.value_counts()[:num].plot(kind='bar',label='Top %s Crimes'%num)
plt.legend()
renum = len(train_df.Category.value_counts()[num:])
train_df.Category.value_counts()[num:].plot(kind='bar',label='Remaining %s Crimes'%renum)
plt.legend()
train_df.PdDistrict.value_counts()
train_df.PdDistrict.value_counts().plot(kind='bar',label='Crimes by PdDistrict')
plt.legend()
ax = train_df.Dates.dt.dayofweek.value_counts().sort_index().plot(kind='bar',label='Crimes by DayOfWeek')
ax.set_xticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.legend(bbox_to_anchor=(1,1.2))
train_df.Resolution.value_counts().plot(kind='bar',label='Resolution')
plt.legend()
mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
plt.imshow(mapdata, cmap = plt.get_cmap('gray'))
#mapdata = np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
#plt.imshow(mapdata, cmap = plt.get_cmap('gray'))
plt.plot(train_df.X,train_df.Y,'bo')
# get rid of bogus lat/long values
train_df = train_df[(train_df.X != -120.5) & (train_df.Y != 90)]
plt.plot(train_df.X,train_df.Y,'bo',label='ALL')
tmp_df = train_df[train_df.Category == 'EMBEZZLEMENT']
plt.plot(tmp_df.X,tmp_df.Y,'go',label='EMBEZZLEMENT')
tmp_df = train_df[train_df.Category == 'BRIBERY']
plt.plot(tmp_df.X,tmp_df.Y,'wo',label='BRIBERY')
plt.legend(frameon=True,numpoints=1,bbox_to_anchor=(1.5,1))
# add columns which may be useful
train_df['Year'] = train_df.Dates.dt.year
train_df['Hour'] = train_df.Dates.dt.hour
train_df['Month'] = train_df.Dates.dt.month
train_df['DayOfMonth'] = train_df.Dates.dt.day
train_df['DayOfWeekInt'] = train_df.Dates.dt.dayofweek
train_df.columns
train_df.Year.value_counts().sort_index().plot(kind='bar',label='Crimes by Year')
plt.legend()
plt.legend(bbox_to_anchor=(1,1.2))
train_df.Month.value_counts().sort_index().plot(kind='bar',label='Crimes by Month')
plt.legend()
plt.legend(bbox_to_anchor=(1,1.2))
train_df.DayOfMonth.value_counts().sort_index().plot(kind='bar',label='Crimes by DayOfMonth')
plt.legend()
plt.legend(bbox_to_anchor=(1,1.2))
train_df.Hour.value_counts().sort_index().plot(kind='bar',label='Crimes by Hour (Of Day)')
plt.legend(bbox_to_anchor=(1,1.2))
train_df.isnull().values.any()
# test dataset
test_df = pd.read_csv("../input/test.csv",parse_dates=['Dates'])
test_df.shape
test_df.columns
test_df.describe(include='all')
plt.plot(test_df.X,test_df.Y,'bo')
# get rid of bogus lat/long values
test_df = test_df[(test_df.X != -120.5) & (test_df.Y != 90)]
plt.plot(test_df.X,test_df.Y,'bo')
test_df.isnull().values.any()


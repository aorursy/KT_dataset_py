# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
usAccidents = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")
print(usAccidents.shape)

print(usAccidents.columns)
usAccidents['newStartTime'] = pd.to_datetime(usAccidents.Start_Time, format='%Y-%m-%d %H:%M:%S')

usAccidents['newEndTime'] = pd.to_datetime(usAccidents.End_Time, format='%Y-%m-%d %H:%M:%S')



usAccidents['STime'] = pd.to_datetime(usAccidents.newStartTime, format='%H:%M:%S')

usAccidents['ETime'] = pd.to_datetime(usAccidents.newEndTime, format='%H:%M:%S')



usAccidents[['newStartTime','newEndTime', 'STime', 'ETime' ]].head()
usAccidents['StartHour'] = usAccidents['newStartTime'].apply(lambda x: x.time().hour)

usAccidents['EndHour'] = usAccidents['newEndTime'].apply(lambda x: x.time().hour)
usAccidents['StartHour'].hist(bins=23)

plt.ylabel('Frequency')

plt.xlabel('Hour of Day')

plt.title('Distribution of accidents over the day')
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.distplot(usAccidents['StartHour'], kde= True, color='red', bins=23)

plt.ylabel('Frequency')

plt.xlabel('Hour of Day')

plt.title('Distribution of accidents over the day')
usAccidents['StartHour'].value_counts()
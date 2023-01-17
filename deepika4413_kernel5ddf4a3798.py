# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib.inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
offense_df = pd.read_csv("../input/crimes-in-boston/offense_codes.csv", encoding = "ISO-8859-1")

offense_df.head()
offense_df.head()

offense_df.describe()
crime_df= pd.read_csv("../input/crimes-in-boston/crime.csv", encoding = "ISO-8859-1")

crime_df.head()
crime_df.describe()
crime_df.isnull().sum()
crime_df.info()
crime_df['OFFENSE_CODE_GROUP'].unique()
min(crime_df['YEAR'])
max(crime_df['YEAR'])
crime_df[crime_df['OFFENSE_CODE'] == 619].count()
crime_df.head()
fig, ax = plt.subplots(figsize=(20,7))

crime_df.groupby(['YEAR','DAY_OF_WEEK']).count()['OFFENSE_CODE'].unstack().plot(ax=ax)

ax.set_xlabel('Year')

ax.set_ylabel('Number of crimes reported')
# Vizualizing the count of each offense group to see the maximum number of offense occuring



crime_df['OFFENSE_CODE_GROUP'].value_counts().plot(kind='bar',

                                    figsize=(20,8),

                                    title="count of each offense group")
# Explore the crimes according to hour of the day.

df1 = pd.DataFrame(crime_df.groupby(['HOUR']).count()['OFFENSE_CODE'])



df1.plot(kind='bar',figsize=(20,8),title="count of each offense group")





# The below graph shows that the Crimes are reported LOW during 1.00 AM till 7.00 AM in the morning and then the reported crime seems 

# to gradually increase



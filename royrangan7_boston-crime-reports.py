# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = "../input/crime.csv"

#rows = 10000;



df = pd.read_csv(path, encoding = "ISO-8859-1")

del df['SHOOTING']



# 1. In Which Places crimes are most common?

df['DISTRICT'].fillna('B2',inplace=True)

DISTRICT = df['DISTRICT'].value_counts(sort = True)

plt.figure(figsize=(20,10))

sns.barplot(DISTRICT.index, DISTRICT.values, alpha=0.8)

plt.title('Districts Where Crimes Are Most Common', fontsize=40)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('DISTRICT', fontsize=12)

plt.show()





# 2. What types of crimes are most common?

OFFENSE_CODE_GROUP = df['OFFENSE_CODE_GROUP'].value_counts(sort = True).nlargest(8)

plt.figure(figsize=(20,10))

sns.barplot(OFFENSE_CODE_GROUP.index, OFFENSE_CODE_GROUP.values, alpha=0.8)

plt.title('Crimes those are Most Common', fontsize=30)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('OFFENSE CODE GROUP', fontsize=12)

plt.show()



# 3. Frequency of crimes change over the Hour?

sns.catplot(x='HOUR',

           kind='count',

            height=8.27, 

            aspect=2,

           data=df)

plt.xticks(size=30)

plt.yticks(size=30)

plt.title('Crimes Count in Hour', fontsize=30)

plt.xlabel('Hour', fontsize=30)

plt.ylabel('Count', fontsize=30)





# 4. Frequency of crimes change over the MONTH?

sns.catplot(x='MONTH',

           kind='count',

            height=8.27, 

            aspect=2,

           data=df)

plt.xticks(size=30)

plt.yticks(size=30)

plt.title('Crimes Count in Month', fontsize=30)

plt.xlabel('MONTH', fontsize=30)

plt.ylabel('Count', fontsize=30)





# 5. Frequency of crimes change over the Day?

days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

sns.catplot(x='DAY_OF_WEEK',

           kind='count',

            height=8.27, 

            aspect=2,

           data=df)

plt.xticks(np.arange(7), days, size=30)

plt.yticks(size=30)

plt.title('Crimes Count Of DAYS', fontsize=30)

plt.xlabel('DAYS', fontsize=30)

plt.ylabel('Count', fontsize=30)



# 6. Frequency of crimes change over the Year?

sns.catplot(x='YEAR',

           kind='count',

            height=8.27, 

            aspect=2,

           data=df)

plt.xticks(size=30)

plt.yticks(size=30)

plt.title('Crimes Count Of YEAR', fontsize=30)

plt.xlabel('YEAR', fontsize=30)

plt.ylabel('Count', fontsize=30)



# 7. Frequency of crimes on Which Street?



STREET = df['STREET'].value_counts(sort = True).nlargest(11)

plt.figure(figsize=(20,10))

sns.barplot(STREET.index, STREET.values, alpha=0.8)

plt.title('Streets where Crimes Mostly Occured', fontsize=30)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('STREET', fontsize=12)

plt.show()



# 8. Frequency of crimes change over Uniform Crime Reports?

sns.catplot(x='UCR_PART',

           kind='count',

            height=8.27, 

            aspect=2,

           data=df)

plt.xticks(size=30)

plt.yticks(size=30)

plt.title('Uniform Crime Report Chart', fontsize=30)

plt.xlabel('UCR PART', fontsize=30)

plt.ylabel('Count', fontsize=30)





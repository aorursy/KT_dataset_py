# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
schools = pd.read_csv('../input/Public_Schools.csv')
schools.head()
schools.info()
schools.isnull().any()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# plot schools per city
schools_per_city = schools['CITY'].value_counts()

sns.set()
plt.rcParams['figure.figsize'] = [20, 7]
sns.barplot(x=schools_per_city.index, y=schools_per_city.get_values())
# plot schools per zipcode
school_zipcode = schools['ZIPCODE'].value_counts()
sns.set()
sns.barplot(x=school_zipcode.index, y=school_zipcode.get_values())
# plot schools per school type.
school_type = schools['SCH_TYPE'].value_counts()
sns.set()
sns.barplot(x=school_type.index, y=school_type.get_values())
schools['PL'].value_counts()
schools_groupby = schools.groupby(['CITY', 'SCH_TYPE'], as_index=False)['X'].count()

sns.set()
sns.factorplot(x='CITY', y='X', hue='SCH_TYPE', data=schools_groupby, kind='bar', size=10.5, aspect=1.4)
# schools_groupby_zipcode
schools_groupby_zipcode = schools.groupby(['ZIPCODE', 'CITY'], as_index=False)['X'].count()

# plot'em.
sns.set()
sns.factorplot(x='ZIPCODE', y='X', hue='CITY', data=schools_groupby_zipcode, kind='bar', size=10.5, aspect=1.4)
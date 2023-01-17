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
df= pd.read_csv('../input/police/police.csv')
df.head()
df.isnull().sum()
df.head()
df.drop('search_type',axis=1,inplace=True)

df.columns
gender=df['driver_gender'].value_counts()
gender

age= df['driver_age'].value_counts()
age

age_1 = df['driver_age'].value_counts().rename_axis('Age').reset_index(name='No of Violations')
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
age_1
age_1.sort_values("Age", axis = 0, ascending = True, 
                 inplace = True, na_position ='last')
age_1
plt.figure(figsize=(30,6))
sns.barplot(x="Age",y="No of Violations",data=age_1)


violation=df['violation'].value_counts()
violation=df['violation'].value_counts().rename_axis('Violation').reset_index(name='No of Violations')
violation
chart=sns.barplot(x="Violation",y="No of Violations",data=violation)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

df.groupby("driver_gender").violation.value_counts()
df['stop_outcome'].value_counts()
df[df.stop_outcome=='Citation'].driver_gender.value_counts()
df[df.stop_outcome=='Warning'].driver_gender.value_counts()
df[df.stop_outcome=='Arrest Driver'].driver_gender.value_counts()
df[df.stop_outcome=='N/D'].driver_gender.value_counts()
df[df.stop_outcome=='No Action'].driver_gender.value_counts()
df[df.stop_outcome=='Arrest Passenger'].driver_gender.value_counts()
df.head()
df.stop_duration.value_counts(dropna = False)

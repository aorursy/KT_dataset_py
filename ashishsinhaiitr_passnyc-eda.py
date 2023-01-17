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
schools=pd.read_csv('../input/2016 School Explorer.csv')
register=pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
schools.head()
register.head()
schools.shape
register.shape
schools.isnull().sum()
schools.dtypes
schools.dtypes.value_counts()
register.isnull().sum()
register.dtypes.value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,10))
#sns.countplot(schools['Grade 8 ELA'])
#schools['Grades']
sns.distplot(schools['Grade 7 Math - All Students Tested'],kde=True)
plt.figure(figsize=(20,10))
schools['Grade 8 Math 4s - American Indian or Alaska Native'].value_counts()
plt.figure(figsize=(20,10))
sns.distplot(schools['Grade 8 ELA - All Students Tested'])







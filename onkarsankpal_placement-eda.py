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
#Importing libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum().sort_values(ascending = False)
#As the person is not placed,so the salary will be considered as 0



df=df.fillna(0)
df.isnull().sum()
def counter_plot(feature):

    sns.countplot(x=feature,data=df)
counter_plot('gender'),df['gender'].value_counts()
counter_plot('ssc_b'),df['ssc_b'].value_counts()
counter_plot('hsc_b'),df['hsc_b'].value_counts()
counter_plot('hsc_s'),df['hsc_s'].value_counts()
counter_plot('degree_t'),df['degree_t'].value_counts()
counter_plot('workex'),df['workex'].value_counts()
counter_plot('specialisation'),df['specialisation'].value_counts()
counter_plot('status'),df['status'].value_counts()
sns.heatmap(df.corr(),annot=True)
def plot_wrt_status(feature):

    sns.countplot(x=feature,hue='status',data=df)
plot_wrt_status('gender')
plot_wrt_status('ssc_b')
plot_wrt_status('hsc_b')
plot_wrt_status('hsc_s')
plot_wrt_status('degree_t')
plot_wrt_status('workex')
plot_wrt_status('specialisation')
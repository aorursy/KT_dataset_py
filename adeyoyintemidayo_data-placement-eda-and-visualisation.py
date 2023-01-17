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
from matplotlib import pyplot as plt

import seaborn as sns 

sns.set(style="whitegrid")
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()

data["salary"] = data["salary"].fillna(0)
data.columns
data.info()
data.describe()
data1 = sns.pairplot(data)

data1
sns.countplot(data['status'],hue=data['gender'])
sns.countplot(data["status"], hue=data["hsc_b"])
sns.catplot(x="status", y="ssc_p", jitter = False,data=data)

sns.catplot(x="status", y="hsc_p", jitter = False,data=data)

sns.catplot(x="status", y="degree_p", jitter = False,data=data)

sns.catplot(x="status", y="mba_p", jitter = False,data=data)
sns.countplot(x = "specialisation", data=data, hue="gender")
data["specialisation"].value_counts()
sns.countplot(x="degree_t" , data = data, hue="gender")
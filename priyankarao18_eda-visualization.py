# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
placement = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
placement.head()
placement.isnull().sum()
placement.info()
sns.countplot(data=placement , x ="gender")
plt.show()
sns.countplot(data=placement , x ="hsc_s")
plt.show()
sns.countplot(data=placement , x = "degree_t")
plt.show()
sns.countplot(data=placement , x = "specialisation")
plt.show()
placement.status.value_counts(normalize = True)
sns.pairplot(data = placement , vars = ["degree_p" , "etest_p" , "mba_p" , "ssc_p" , "hsc_p"])
plt.show()
plt.figure(figsize = [15,12])
plt.subplot(3,2,1)
sns.boxplot(data=placement , x ="status" , y ="mba_p")
plt.subplot(3,2,2)
sns.boxplot(data=placement , x ="status" , y ="etest_p")
plt.subplot(3,2,3)
sns.boxplot(data=placement , x ="status" , y ="degree_p")
plt.subplot(3,2,4)
sns.boxplot(data=placement , x ="status" , y ="hsc_p")
plt.subplot(3,2,5)
sns.boxplot(data=placement , x ="status" , y ="ssc_p")
plt.show()
sns.countplot(x = "specialisation" , hue = "status" , data =placement)
plt.show()
sns.countplot(x = "workex" , hue = "status" , data =placement)
plt.show()
sns.countplot(x = "degree_t" , hue ="specialisation" , data=placement)
plt.show()
sns.countplot(x="hsc_s" , hue = "degree_t" , data=placement)
plt.show()
sns.catplot(data=placement , x = "degree_t" , hue = "workex" , kind="count")
plt.show()
placement.salary.describe()
sns.stripplot(data=placement , x = "workex" , y ="salary" , jitter=True , hue = "specialisation")
plt.show()
sns.scatterplot(data = placement , x = "etest_p" , y = "salary" , hue = "specialisation")
plt.show()
sns.scatterplot(data = placement , x = "mba_p" , y = "salary" , hue="specialisation")
plt.show()

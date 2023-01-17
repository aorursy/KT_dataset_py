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
data1 = pd.read_csv("../input/salaries/Salaries.csv")

data1.head(10)
data1.describe()
data1.salary.plot(kind="hist",bins=10,figsize=(10,10))
plt.show()
sns.barplot(data1["salary"], y="rank",data=data1)
plt.show()
sns.barplot(x="discipline",y="salary",data=data1)
plt.show()
sns.barplot(x="sex",y="salary",data=data1)
plt.show()
sns.scatterplot(data=data1, x="sex",y="salary",hue="discipline")
plt.show()
sns.regplot(x="service",y="salary",data=data1)
plt.show()
sns.boxplot(x="sex",y="salary",hue="discipline",data=data1)
plt.show()

sns.swarmplot(x="sex",y="salary",hue="discipline",data=data1)
plt.show()
sns.pairplot(data1)
plt.show()

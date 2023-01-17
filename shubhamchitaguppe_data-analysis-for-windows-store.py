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
import matplotlib.pyplot as plt
import seaborn as sns
plt.show()
sns.set_style("whitegrid")
%matplotlib inline
data=pd.read_csv("/kaggle/input/windows-store/msft.csv")
data.head()
data["Price"].value_counts()
data.tail(1)
data.info()
data.isnull().sum()
data["Price"]
plt.figure(figsize=(10,10))
sns.pairplot(data,hue="Category")
plt.figure(figsize=(10,20))
sns.jointplot(x="Rating",y="No of people Rated",kind='kde',data=data)
plt.figure(figsize=(20,8))
sns.barplot(x="Category",y="No of people Rated",data=data)
data.fillna(0,inplace=True)
data['Name']
data['Category'].value_counts()
plt.figure(figsize=(25,7))
sns.countplot(x="Category",data=data)
data.corr
plt.figure(figsize=(12,6))
sns.heatmap(data.corr())
plt.figure(figsize=(22,8))
sns.boxplot(x="Rating",y="No of people Rated",data=data,hue="Category")
plt.figure(figsize=(22,8))
sns.swarmplot(x="Category",y="No of people Rated",data=data,hue='Rating')
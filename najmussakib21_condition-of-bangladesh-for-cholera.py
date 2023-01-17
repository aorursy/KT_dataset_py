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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/cholera-dataset/data.csv')

data.head()
data.drop('WHO Region',axis=1,inplace=True)
#selecting only Bangladesh data

bd = data[data['Country']=='Bangladesh']
bd
#checking if there is any missing values or not

bd.isnull().sum()
plt.subplots(figsize=(8,10))

plt.plot(bd["Year"], bd["Number of reported cases of cholera"], linewidth=2)

plt.xlabel("Year")

plt.ylabel("Number of reported cases of cholera")
fig,ax=plt.subplots(figsize=(12,14))

ax.plot(bd['Year'],bd['Number of reported cases of cholera'],color='red')

ax.set_xlabel('year',fontsize=14)

ax.set_ylabel('Number of reported cases of cholera',color='red',fontsize=14)



ax2 = ax.twinx()

ax2.plot(bd['Year'],bd['Number of reported deaths from cholera'],color='blue')

ax2.set_ylabel('Number of reported deaths from cholera',color='blue',fontsize=14)

plt.show()
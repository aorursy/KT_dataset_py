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
# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
crop=pd.read_csv("../input/crop-production-in-india/crop_production.csv")

crop.head()
crop.columns
print(crop.shape)
print(crop.size)
crop.info()
crop['Season'].unique()
plt.figure(figsize = (6,6))

segment = crop['Season'].value_counts()

segment_label = crop['Season'].unique()

color = ('LightPink', "LightBlue" , 'LightGreen','red','green','Gold')



plt.pie(segment,

       autopct = '%1.1f%%',

       labels = segment_label,

       explode = (0.06,0.05,0.05,0.07,0.08,0.05),

       shadow = True,

       colors = color);
sns.catplot(data=crop,x="Crop_Year",aspect=3,kind='count')
crop.describe()
corr=crop[['Area','Production']].corr()

sns.heatmap(corr,annot = True , cmap = 'YlGnBu')
sns.heatmap(crop.isnull(),yticklabels=False)
crop.isnull().sum()
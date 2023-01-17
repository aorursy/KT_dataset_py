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
# all the required imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# importing the dataset as 'ds'

ds = pd.read_csv('../input/married-at-first-sight/mafs.csv')
ds.head()
# number of couples per season

sns.countplot(x="Season",data=ds,palette="GnBu_d",edgecolor="black")



# changing the font size

sns.set(font_scale=1) 
# counting the people of different ages

sns.countplot(x='Age',data=ds,palette='GnBu_d',edgecolor="black")



# changing the font size

sns.set(font_scale=1) 
# resizing the plot for better visibilty

plt.figure(figsize=(10,10))     



# it rotates the x names for better readability

plt.xticks(rotation=90) 



# number of people from each location

sns.countplot(x='Location',data=ds,palette='GnBu_d',edgecolor="black")



# changing the font size

sns.set(font_scale=3)           
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
Data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

Data.info()
Data.head(10)
by_Category  = Data.Category.value_counts()

top_10_Category = by_Category[:10]

top_10_Category
import matplotlib.pyplot as plt

top_10_Category.plot(kind='bar'); #Here ; is used to avoid the message

plt.title('Category With Most Number of Apps')
#To plot horizontally

top_10_Category.plot(kind='barh');

plt.title('Category With Most Number of Apps')


import seaborn as sns



#plt.bar(top_10_Category.index, top_10_Category); #This looks messy

#sns.barplot(top_10_Category.index, top_10_Category); #This looks messy too
sns.barplot(top_10_Category, top_10_Category.index)

plt.title('Category With Most Number of Apps')

plt.xlabel('Number of Apps')

plt.xticks(fontsize=10)

plt.ylabel('Category')

plt.yticks(fontsize=12)
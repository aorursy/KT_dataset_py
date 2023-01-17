# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

data.head()
sns.pairplot(data)

plt.show()
data['Avg marks'] = (data['math score'] + data['reading score'] + data['writing score'])/3

data.head()
pvt_tbl = pd.pivot_table(data = data, index = ["parental level of education"], columns = ["race/ethnicity"], aggfunc = {'Avg marks' : np.mean})

hm = sns.heatmap(data = pvt_tbl, annot = True, cmap = "Greens")

bottom, top = hm.get_ylim()

hm.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
pvt_tbl = pd.pivot_table(data = data, index = ["gender"], columns = ["test preparation course"], aggfunc = {'Avg marks' : np.mean})

hm = sns.heatmap(data = pvt_tbl, annot = True, cmap = "Greens")

bottom, top = hm.get_ylim()

hm.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
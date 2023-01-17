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
data = pd.read_csv('../input/fifa19/data.csv', index_col='Unnamed: 0')

data.head(5)
data[['Preferred Foot']]
data['Preferred Foot'].value_counts()
foot_data = data['Preferred Foot'].value_counts().reset_index()

foot_data
# Now renaming the column index

foot_data.columns = ['Foot', "Total Players"]

foot_data
# setting column=Foot as index

foot_data.set_index('Foot', inplace=True)

foot_data
sns.set_style('darkgrid')

fig = plt.figure(figsize=(5,5))

sns.barplot(x=foot_data.index, y='Total Players', data=foot_data, palette=sns.color_palette('bright'))

plt.title("Preferred Foot", size=20)

plt.xlabel("Foot", size=15)

plt.ylabel("Number of Players", size=15)

plt.show()
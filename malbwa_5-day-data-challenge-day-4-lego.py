# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Create list of data files to read into data structure

data_files = [

    'colors.csv',

    'inventories.csv',

    'inventory_parts.csv',

    'inventory_sets.csv',

    'part_categories.csv',

    'parts.csv',

    'sets.csv',

    'themes.csv'

]



data = {}
# Read csv data files into 'data' dictionary with

# filename as the key as a pandas dataframe

for file in data_files:

    file_path = f'../input/{file}'

    file_nm, file_ext = os.path.splitext(file)

    if file_nm not in data:

        data[file_nm] = pd.read_csv(file_path)
for key in data:

    print(key)
# for key in data:

#     print(key, '\n')

#     print(data[key].describe(), '\n')

#     print('####################\n')
# for key in data:

#     print(key, '\n')

#     print(data[key].head(3), '\n')

#     print('####################\n')
data['sets'].head()
fig, ax = plt.subplots(figsize=(16,4))

sns.countplot(data['sets']['year']).set_title('Lego Sets by Year')



for label in ax.xaxis.get_ticklabels():

    label.set_visible(False)

for label in ax.xaxis.get_ticklabels()[::3]:

    label.set_visible(True)



plt.xticks(rotation=45)

plt.show()
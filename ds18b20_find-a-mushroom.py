# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
import seaborn as sns # optimize visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/mushrooms.csv")
data.shape
data.head()
data.tail()
data.describe()
#Obtain total number of mushrooms for each 'cap-color' (Entire DataFrame)
cap_colors = data['cap-color'].value_counts()
cap_colors_count = cap_colors.values # Provides numerical values //.tolist()
cap_colors_label = cap_colors.index.values
plt.bar(cap_colors_label, cap_colors_count)
plt.show()
class_e_p = pd.concat([data['class'], data['cap-color']], axis=1)
class_e_p.shape
class_e = class_e_p.loc[class_e_p['class']=='e', :]
class_e_count = class_e['cap-color'].value_counts()
class_e_count
class_p = class_e_p.loc[class_e_p['class']=='p', :]
class_p_count = class_p['cap-color'].value_counts()
class_p_count
set(class_e_count.index.tolist())-set(class_p_count.index.tolist())
class_p_count['r'] = 0
class_p_count['u'] = 0
# https://matplotlib.org/gallery/statistics/barchart_demo.html
n_groups = 10

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, class_e_count, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config,
                label='e')

rects2 = ax.bar(index + bar_width, class_p_count, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config,
                label='p')
ax.set_xlabel('Colors')
ax.set_ylabel('Counts')
ax.set_title('Counts by colors and e/p')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(cap_colors_label)
ax.legend()

fig.tight_layout()
plt.show()
describe.loc['freq', :] > data.shape[0] * 0.75
train_data = data.iloc[:, 1:]  # slice train part from colomn_1
train_data.shape
train_data.tail()
train_data.isnull().sum()
train_data = train_data.replace('?', np.nan)  # replace '?' with np.nan
train_data.isnull().sum()
non_nan_ratio = 0.75
# keep columns which Non-NaN count > non_nan_ratio * ALL
# delete columns which Non-NaN count < non_nan_ratio * ALL
# delete columns which NaN count > non_nan_ratio * ALL
train_data.dropna(axis=1, how='any', thresh=train_data.shape[0] * non_nan_ratio, inplace=True)
train_data.describe()
# vectorize strings to numbers
train_data = pd.get_dummies(train_data, dummy_na=False)  # dummy_na=True take NaN as a legal feature label
train_data.shape
data.isnull().sum()
column_name = train_data.columns.values
label_data = data.iloc[:, 0]
label_data.replace({'e': 0, 'p': 1}, inplace=True)
label_data.shape
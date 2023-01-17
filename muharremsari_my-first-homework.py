# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
main_data = pd.read_csv('../input/number_of_births_in_Turkey.csv')
main_data.info()
data = main_data.loc[:,["mother_age_group","year","istanbul","ankara","izmir","diyarbakir","sanliurfa","turkey"]]
filter_under15 = data.mother_age_group=="under15"
data_under15 = data[filter_under15]
data_under15.info()
data_under15.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data_under15.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data_under15.head()
data_under15.tail()
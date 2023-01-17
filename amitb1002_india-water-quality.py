# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
water_df = pd.read_csv("../input/IndiaAffectedWaterQualityAreas.csv", encoding='latin1')
water_df.info()
water_df['State Name'].unique()
maximg = 19
i = 1
for state_name in water_df['State Name'].unique():
    temp = water_df[water_df['State Name'] == state_name]
    tempnew = temp.groupby('Quality Parameter').count()
    sns.barplot(x=tempnew.index, y= tempnew['State Name'], data = tempnew)
    plt.ylabel(state_name)
    plt.figure()
    if i == maximg:
        break
    i = i + 1

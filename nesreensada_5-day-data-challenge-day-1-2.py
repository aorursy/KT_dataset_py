# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Read a csv file 
data = pd.read_csv('../input/millenniumofdata_v3_headlines.csv')
data.describe()
# Any results you write to the current directory are saved as output.

# read the macro economics full file
data_full = pd.read_excel('../input/millenniumofdata_v3_final.xlsx',sheetname='A4. Ind Production 1270-1870',skiprows=7)
data_full.describe()

# extract one numeric column from dataframe to plot hist
data_full.head()
leather = data_full['Leather']
plt.hist(leather)


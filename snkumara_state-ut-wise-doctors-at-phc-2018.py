# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import required libraries

import pandas as pd

import matplotlib.pyplot as plt
#import data set

phc_data = pd.read_excel('/kaggle/input/phc2018/datafile.xls')
phc_data.columns = ["Sl_no","State","Required","Sanctioned","In Position","Vacant","Shortfall"]
phc_data.head()
phc_data.head(10)
phc_data.tail()
phc_data.tail(10)
phc_data.info()
phc_data.isnull().sum()
phc_data.isna().sum()
all_row_max =phc_data.max()

all_row_max
#selecting all the row except last

phc_data = phc_data[:-1]

phc_data.tail()
phc_data.fillna(0, inplace=True)
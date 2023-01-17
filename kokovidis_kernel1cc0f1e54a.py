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

        

        

pd.set_option('display.max_columns', 500) # show all columns

pd.options.display.max_rows = 200 # show 200 rows

        

import warnings

warnings.filterwarnings("ignore") #remove warning messages during csv import



# Any results you write to the current directory are saved as output.
raw_data_original = pd.read_csv("/kaggle/input/thecarconnection32000/fullspecs (1).csv", nrows=110, index_col=0).transpose()

raw_data_original['Full Name'] = raw_data_original.index

raw_data_original
raw_data_original.to_csv('fullspecs.csv')
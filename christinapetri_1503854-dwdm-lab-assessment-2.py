# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data_path = '../input/credit-card-data.csv' # Path to data file

data = pd.read_csv(data_path)
data.head(15)
# What columns are in the data set ? Do they have spaces that I should consider

data.columns
data.describe()
for col in data.columns[2:]:
    data[col].plot(kind='bar')
    plt.title('Bar Plot for '+col)
    plt.show()

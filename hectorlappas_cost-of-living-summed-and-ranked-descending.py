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
pd.set_option('max_rows',160)

cost_data = pd.read_csv('/kaggle/input/cost-of-living/cost-of-living.csv',index_col=[0])

cost_data.drop(index="Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate")

cost_data_sum =cost_data.sum(axis=0)

cost_data_sum.rank(axis=0,ascending=True,method='max')



print(cost_data_sum.sort_values(ascending=False))
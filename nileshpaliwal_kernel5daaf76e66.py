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
data = pd.read_csv('/kaggle/input/mercedesbenz-greener-manufacturing/train.csv')
data.head()
data.dtypes
# run through each columns and check the variance of the column having type = int64 or float64 equals to zero

# and then removing those columns from the data.

for i in data.columns:

    if data[i].dtype in ['float64' , 'int64']:

        if data[i].var()==0:

            data.drop(i,axis=1,inplace=True)
#columns with variance equals to 0 removed.

data.head()
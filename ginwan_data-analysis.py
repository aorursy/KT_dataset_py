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
import pandas as pd

amazon=pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv",encoding="latin1",header=None)
amazon.head(5)
amazon.tail(10)
headers=["year","state","month","number","date"]

print(headers)
amazon.columns=headers
amazon.head(10)
amazon.dropna(subset=["number"],axis=0)
print(amazon.columns)
amazon.dtypes
amazon.describe()
amazon.describe(include="all")
amazon[["year","number"]].describe()
amazon.info
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
s = pd.Series([1, 3, 5, np.nan, 6, 8])

s
dates = pd.date_range('20130101', periods = 6)

dates
data_set = pd.read_csv("../input/Pokemon.csv")

data_set
data_set.dtypes
data_set.head(20)
data_set.index
data_set.columns
data_set['Name']
data_set.describe()
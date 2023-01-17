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
# Series - 1D labeled homogeneously-typed array

# DataFrame - general 2D labeled, size-mutable tabular structure with 

# potentially heterogeneously-typed columns



s = pd.Series([1, 3, 5, np.nan, 6, 8])

s
dates = pd.date_range('2013', periods = 6)

dates
# 6 rows, 4 columns

df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list('ABCD'))

df
data_set = pd.read_csv("../input/anime-recommendations-database/anime.csv")

data_set
data_set.index
data_set.columns
data_set['name']
#If data in dataset contains dates, description is different

data_set.describe()
s = pd.Series(['a', 'a', 'b', 'c']) #categorical

s.describe()
s = pd.Series([np.datetime64("2000-01-01"),

               np.datetime64("2000-01-02"),

               np.datetime64("2000-01-03"),

              ]) # Categorical

s.describe()
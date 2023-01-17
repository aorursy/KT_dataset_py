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
data = pd.read_csv("../input/athlete_events.csv")
data.columns
data.shape
data.info()
data.head()
data.tail()
print((data.Age).value_counts(dropna = False))
data.describe()
data.boxplot(column = "Age")
data_tidy = data.head()

data_tidy
melted = pd.melt(frame = data_tidy, id_vars = "Name", value_vars = ["Weight","Height"])

melted
melted.pivot(index = "Name", columns = "variable", values = "value")
data1 = data.head()

data2 = data.tail()

conc_row = pd.concat([data1,data2], axis = 0, ignore_index = True)

conc_row
data3 = data.Name.tail()

data4 = data.Age.tail()

conc_col = pd.concat([data3,data4], axis = 1)

conc_col
data.dtypes
data.Name = data.Name.astype('category')

data.Year = data.Year.astype('float')
data.dtypes
data.info()
data.Medal.value_counts(dropna = False)
data.Medal.dropna(inplace = True)
assert data.Medal.notnull().all()
data.Medal.value_counts(dropna = False)
data.Age.value_counts(dropna = False)
data.Age.describe()
data.Age.fillna(float(26), inplace = True)
assert data.Age.notnull().all()
data.Age.value_counts(dropna = False)
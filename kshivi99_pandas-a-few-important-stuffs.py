# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/bengaluru-house-price-data/Bengaluru_House_Data.csv")

data.head()
data.area_type
data['area_type']
# Selecting a row



data.iloc[0]
# iloc is row first, column second.

# to get a column - : means all rows, after comma the column number.



data.iloc[:, 0]
# Now the above syntax can be used with many combinations.

# to select data for first three row for 1st column - data.iloc[:3, 0]

# to select 2nd and 3rd row for all columns



data.iloc[1:3, :]
# its also possible to pass a list.



data.iloc[[1, 2, 4] , 0]
# its also possible to select from the end



data.iloc[-5:]
# loc follows the same principle as iloc, but you can index via strings.



data.loc[:, ['area_type', 'availability']]
data.availability == "Ready To Move"
# now this true/false could abe used in conjunction with loc to get relevant rows.



data.loc[data.availability == "Ready To Move"]
# we can combine different conditional statements.



data.loc[(data.availability == "Ready To Move") | (data.size == "2 BHK")]
# using isin() instead of multiple conditional statements.



data.loc[data['area_type'].isin(["Plot Area", "Built-up Area"])]
# isnull() and notnull()



print(data.loc[data['size'].notnull()])
data.describe()
data.bath.mean()
data.area_type.unique()
data.area_type.value_counts()
data.sort_values(by='price')
# by default, sorting happens in ascending order. To do it in descending.



data.sort_values(by='price', ascending=False)
data.bath.dtype
# use astype() to convert a data type from one to another.
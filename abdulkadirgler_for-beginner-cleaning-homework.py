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
data = pd.read_csv("/kaggle/input/fifa19/data.csv")
data.head()
data.tail()
data.shape
data.info()
data.columns
print(data["Nationality"].value_counts(dropna=False))
data.describe()
data1 = data.head(50)
data1.boxplot(column="Age",by="Overall")
data_new = data.head()
data_new
melted = pd.melt(frame=data_new,id_vars="Name",value_vars=["GKHandling","GKReflexes"],value_name="capability_level")
melted
melted.pivot(index="Name",columns="variable",values="capability_level")
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row
data1 = data["SlidingTackle"].head()
data2 = data["Marking"].head()
conc_data_col = pd.concat([data1,data2],axis=1)
conc_data_col
data.dtypes
data["Age"] = data["Age"].astype("float")
data["Photo"] = data["Photo"].astype("category")
data.info()
data["Age"].value_counts(dropna=True)
data["Age"].dropna(inplace=True)
assert data["Age"].notnull().all()
data["Age"].fillna("empty",inplace=True)
assert data["Age"].notnull().all()
# for example
assert data.columns[2] == "Name"
assert data.Age.dtypes == np.float
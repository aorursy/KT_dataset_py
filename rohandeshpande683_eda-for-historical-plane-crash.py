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
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset and setting the index as date, so it will be easier to filter the data
df = pd.read_csv("../input/planecrashinfo_20181121001952.csv", index_col=['date'], parse_dates=True)
print(df)
#inspecting the dataset using shape
df.shape
#inspecting the dataset by columns
df.columns
#checking the data types and null values in the columns if any
df.info()
#check the statistical information about the dataset
df.describe()
#checking the count of values for some import columns
print(df["location"].value_counts(dropna=False))
#checking the percentage of accidents from a specific localtion
print(df.location.value_counts(normalize=True)*100)
#as you can see from the percentage columns no location contributes significantly to the accidents
#checking the count of the operator for quantity
print(df['operator'].value_counts(dropna=False))
#checking the percentage of the operator column
print(df.operator.value_counts(normalize=True)*100)
#as you can see the Aeroflot and Military - U.S Air Force contributes to just 4 & 3 percent
df.loc['2018']




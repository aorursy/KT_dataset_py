# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

pro_no_df = pd.read_csv("../input/procurement-notices.csv")
pro_no_df.head()
len(pro_no_df)

pro_no_df["Country Name"].unique()
len(pro_no_df["Country Name"].unique())

import matplotlib.pyplot as plt
pro_no_df["Country Name"].value_counts().plot(kind = 'bar', figsize = (18,8))

len(pro_no_df["Major Sector"].unique())
pro_no_df["URL"][0]
pro_no_df["Notice Type"].unique()

pro_no_df["Procurement Type"].unique()
pro_no_df["Procurement Type"].value_counts(dropna=False).plot(kind = 'bar', figsize = (18,8))
# Any results you write to the current directory are saved as output.

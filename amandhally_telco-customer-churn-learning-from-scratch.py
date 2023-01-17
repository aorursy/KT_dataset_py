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
#Creating File path for CSV FIle
myCsvFile = "../input/WA_Fn-UseC_-Telco-Customer-Churn.csv"
# Loading CSV file to Pandas
df = pd.read_csv(myCsvFile)
#Print the Object Type
type(df)
# Let's check if we have the data right by checking it's first 5 Rows
df.head(5)
# Let's check the last enteries in too using tail 
df.tail(5)

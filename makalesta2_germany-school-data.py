# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import missingno as msno

# Any results you write to the current directory are saved as output.
germany=pd.read_excel("../input/SchoolsGermany.xlsx")
germany=pd.DataFrame(germany)

germany.head()
germany.nunique()
germany.shape
msno.bar(germany,color="red");
msno.matrix(germany);
germany.isnull().sum()
values_count=germany.isnull().sum()
total_cells=np.product(germany.shape)

total_missing=values_count.sum()

print("Percentage of missing data:",(total_missing/total_cells)*100)
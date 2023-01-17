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
data = pd.read_csv("../input/globalterrorismdb_0718dist.csv", encoding='iso-8859-1')
#Database describe. 181691 Row Count. Basic info 
data.describe()

#Basic info about database. 
#181691 entry, 135 Columns, 55 Float, 22 int, 58 Object datatype column.
data.info()
data.head()
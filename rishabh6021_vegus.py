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
import pandas as pd
vegus=pd.read_csv("../input/vegetarian_restaurants_US_datafiniti.csv")
vegus.columns
vegus.dtypes
vegus.shape
vegus.head()
vegus.describe()
vegus.info()
veg=vegus[['categories','city','cuisines','postalCode','province']].copy()
veg.head()
veg.dropna(subset=['cuisines'],inplace=True)
veg['cuisines'].str.contains('Veg').value_counts()[True]
veg[veg['cuisines'].str.contains("Veg") == True]
veg.head()
veg.shape
veg['city'].value_counts()

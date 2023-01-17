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
data = pd.read_excel('/kaggle/input/superstore/US Superstore data.xls')
data.info()
data.head()
data.isnull().sum()
(data.groupby(['Product Name'])['Profit'].sum()/data.groupby(['Product Name'])['Product Name'].count()).head()
data.groupby(['City'])['Order ID'].count().reset_index().sort_values(by='Order ID',ascending = False).head()
data.groupby(['State'])['Order ID'].count().reset_index().sort_values(by='Order ID',ascending = False).head()
data.groupby(['Product Name'])['Discount'].describe().reset_index().sort_values(by = 'max', ascending = False).head()
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('../input/finance-accounting-courses-udemy-13k-course/udemy_output_All_Finance__Accounting_p1_p626.csv')
data.head()
import seaborn as sns
data.isnull().sum()
data.shape
data=data.dropna(how='all')
data.shape
sns.heatmap(data.isnull())
#rows having missing price deatails
data[:][data['price_detail__amount'].isna()==True]
data=data.drop(data[:][data['price_detail__amount'].isna()==True].index)
data.shape
data.head()
data.isnull().sum()
data.fillna('0',inplace=True)
data.shape
data.isnull().sum()
data.head()
data[:][data['discount_price__price_string']=='0']
data[:][data['is_paid']=='False']
data[:][data['num_subscribers']>100000]
data[:][data['is_wishlisted']==True]

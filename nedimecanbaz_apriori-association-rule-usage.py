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
import seaborn as sns
!pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv("../input/supermarket/GroceryStoreDataSet.csv",names=['Products'],sep=',')
df.head()
df.shape
df.describe().T
data = list(df["Products"].apply(lambda x:x.split(',')))
data
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df = pd.DataFrame(te_data,columns=te.columns_)
df
freg_items=apriori(df,min_support=0.2,use_colnames=True,verbose=1)
#the support value in the first column is the probability that the product (s) specified in column 2 will appear in all sales(the time when the data was received is not specified)
#based on taken data a time in a day,the frequencies can be changed due to sales behaviour will change
freg_items
freg_items.sort_values(by="support",ascending=False)
df1=association_rules(freg_items,metric="confidence",min_threshold=0.4)
df1
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
!pip install mlxtend

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
df = pd.read_csv('/kaggle/input/supermarket/GroceryStoreDataSet.csv',names=['Products'],header=None)
 #replacing 'cock'with 'coke' and 'suger' with 'sugar' because they are written wrongly :)
df.replace(to_replace='COCK', value='COKE', regex=True, inplace=True)
df.replace(to_replace='SUGER', value='SUGAR', regex=True, inplace=True)
df.values
data = list(df["Products"].apply(lambda x:x.split(',')))
data
from mlxtend.preprocessing import TransactionEncoder
#All the products seen in the all transactions will be put as columns(variables)
#For all transactions, we will put 'true' to the bought products. (to the corresponding product column) and 
#'false' to the rest of the items which are not bought in that transaction. 
#then this pandas series is converted to dataframe df2:
te = TransactionEncoder()
te_data = te.fit(data).transform(data)
df2 = pd.DataFrame(te_data,columns=te.columns_)
df2
from mlxtend.frequent_patterns import apriori
df3 = apriori(df2,min_support=0.15,use_colnames=True, verbose=1)
df3
df3.sort_values(by="support", ascending=False)
#We can see from the table above that the most freqently bought product is Bread with its 0.65 support value.
AR = association_rules(df3, metric = "confidence",min_threshold = 0.6)
AR
AR.sort_values(by="support", ascending=False)
AR.sort_values(by="confidence", ascending=False)
AR.sort_values(by="lift", ascending=False)
df_filt = AR[ (AR["support"] >= 0.15) &  (AR["confidence"] >= 0.75 )]
df_filt
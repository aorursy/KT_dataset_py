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
data = pd.read_csv('/kaggle/input/onlineretail/OnlineRetail.csv',encoding='latin1')
data.head()
data.shape
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

data['InvoiceDate'] = (data['InvoiceDate'] - data['InvoiceDate'][0])

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

data['InvoiceDate'] = data['InvoiceDate'].dt.minute
data['InvoiceDate_diff'] = data.InvoiceDate.diff().fillna(0)
diff_min = data.groupby('Description').mean().abs()

count  = data.groupby('Description').count()
diff_min['counts'] = count.Quantity
diff_min = diff_min.sort_values(['InvoiceDate_diff','counts'],ascending=[True, False])

diff_min['monetary_earn_rate'] = diff_min.Quantity * diff_min.UnitPrice * diff_min.counts / (diff_min.InvoiceDate_diff + 0.01)
import matplotlib.pyplot as plt

plt.plot(sorted(diff_min.monetary_earn_rate))

plt.ylabel('some numbers')

plt.show()



diff_min = diff_min.sort_values('monetary_earn_rate',ascending = False)
import seaborn as sns

ax = sns.barplot(x="monetary_earn_rate", y=diff_min.head(15).index, data=diff_min.head(15))
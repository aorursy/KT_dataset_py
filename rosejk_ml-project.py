# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_trainData = pd.read_csv('../input/train.csv')

df_item=  pd.read_csv('../input/items.csv')

df_oil = pd.read_csv('../input/oil.csv')

df_stores = pd.read_csv('../input/stores.csv')

df_transactions = pd.read_csv('../input/transactions.csv')
df_trainData.head(10)
df_item.head(10)
df_oil.head(10)
df_transactions.head(10)
data
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
properties = pd.read_csv('../input/properties_2016.csv')

train = pd.read_csv('../input/train_2016_v2.csv')

dd = pd.read_excel('../input/zillow_data_dictionary.xlsx')
train.head(1)
df = pd.merge(train, properties, how='left', left_on='parcelid', right_on='parcelid')

df.shape
df['transactiondate'] = pd.to_datetime(df['transactiondate'])

df['transactiondate'].dtypes
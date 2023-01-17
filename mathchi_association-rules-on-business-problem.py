# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from mlxtend.frequent_patterns import apriori, association_rules



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/online-retail-data-set-from-ml-repository/retail_dataset.csv', sep=',')

df.head()
df.shape
items = (df['0'].unique())

items
encoded_vals = []

for index, row in df.iterrows(): 

    labels = {}

    uncommons = list(set(items) - set(row))

    commons = list(set(items).intersection(row))

    for uc in uncommons:

        labels[uc] = 0

    for com in commons:

        labels[com] = 1

    encoded_vals.append(labels)
ohe_df = pd.DataFrame(encoded_vals)
ohe_df
freq_items = apriori(ohe_df, min_support = 0.2, use_colnames = True, verbose = 1)
freq_items.head()
association_rules(freq_items, metric = "confidence", min_threshold = 0.6)
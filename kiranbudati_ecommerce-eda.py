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
import matplotlib.pyplot as plt
oct_data= pd.read_csv("/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Oct.csv")
oct_data.info()
oct_data.head()
oct_data.columns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

axes[0].bar(oct_data["event_type"].value_counts().index,oct_data["event_type"].value_counts().values)

axes[1].pie(oct_data["event_type"].value_counts().values)

axes[1].legend(oct_data["event_type"].value_counts().index)

print('Total Unique Brands : {} '.format(len(oct_data['brand'].unique())))
event_by_brand = oct_data.groupby("event_type")['brand']
event_by_brand.head(10)
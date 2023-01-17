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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/train.tsv', sep='\t')
train.head()
train.info()
train.describe()
plt.hist(train.item_condition_id,bins=np.arange(1,7)-0.5)
plt.hist(train.price,bins=np.arange(0,2000,200))
plt.hist(train.shipping,bins=np.arange(3)-0.5)
cn_count = train.category_name.str.count("/")+1

cn_count.describe()
train[cn_count>3].head()
train.brand_name.drop_duplicates().count()
train.groupby(by="brand_name")["train_id"].count().sort_values(ascending=False).head(20)
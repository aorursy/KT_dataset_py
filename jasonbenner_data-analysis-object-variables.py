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

train_df = pd.read_csv("../input/train.csv")
train_df.columns
train_df.info()
train_df.full_sq.value_counts(dropna=False)
train_df.full_sq.plot('hist')
train_df.life_sq.plot('hist')
train_df[train_df.life_sq < 10]
train_df_objects = train_df.select_dtypes(include = [np.object])
train_df_objects.columns
train_df.product_type_categorical = train_df_objects.product_type.astype('category')
train_df.product_type_categorical
df_product_type = pd.get_dummies(train_df_objects.product_type_categorical)
df_product_type
train_df = train_df.join(df_product_type)
train_df
train_df = train_df.rename(columns = {'Investment': 'product_type_investment', 'OwnerOccupier':'product_type_owner_occupier'})
train_df.drop('product_type', axis =1)
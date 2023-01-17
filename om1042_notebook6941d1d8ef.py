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
# This Python 3 environment comes with many helpful analytics libraries installed

base_dir= '../input/'

df_aisles = pd.read_csv(base_dir+'aisles.csv')

df_aisles.head(5)
df_departments = pd.read_csv(base_dir+'departments.csv')

df_departments.head(5)
df_prior_order = pd.read_csv(base_dir+'order_products__prior.csv')

df_prior_order.head(5)
df_orders = pd.read_csv(base_dir+'orders.csv')

df_orders.head(5)
df_products = pd.read_csv(base_dir+'products.csv')

df_products.head(5)
df_submission = pd.read_csv(base_dir+'sample_submission.csv')

df_submission.head(5)
df_order_products__train = pd.read_csv(base_dir+'order_products__train.csv')

df_order_products__train.head(5)
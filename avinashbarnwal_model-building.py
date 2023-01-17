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
odelpd.set_option('display.width', 1000)

pd.set_option('display.max_columns', 500)



def load_csv(file_name):

  return pd.read_csv("../input/" + file_name)



orders = load_csv("orders.csv")

products = load_csv('products.csv')

order_products_train = load_csv('order_products__train.csv')

order_products_prior = load_csv('order_products__prior.csv')

aisles = load_csv('aisles.csv')

departments = load_csv('departments.csv')



user_group=orders.groupby(['user_id']).count()
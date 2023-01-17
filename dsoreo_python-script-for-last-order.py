# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

import itertools as itr



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
aisles = pd.read_csv('../input/aisles.csv')

det = pd.read_csv('../input/departments.csv')

opp = pd.read_csv('../input/order_products__prior.csv')

opt = pd.read_csv('../input/order_products__train.csv')

orders = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')

ss = pd.read_csv('../input/sample_submission.csv')
#Write serial number to maintain order of orders!

opp["sno"] = pd.Series(np.arange(len(opp)))

opt["sno"] = pd.Series(np.arange(len(opp)))

orders["sno"] = pd.Series(np.arange(len(opp)))



prior_orders = orders[orders.eval_set == "prior"]

train_orders = orders[orders.eval_set == "train"]

test_orders = orders[orders.eval_set == "test"]



#Get the latest prior order

prior_test_orders = prior_orders[prior_orders.user_id.isin(test_orders.user_id)]

prior_test_max = prior_test_orders[prior_test_orders.groupby(['user_id'])['sno'].transform(max) == prior_test_orders['sno']]



orders_with_products  = pd.merge(prior_test_max,opp,on=["order_id"],how="left")



output_file = test_orders.copy()

op_file_unformatted = pd.merge(output_file,orders_with_products, on=["user_id"],how="left")



#Write prior order in submission file. The output file needs final processing in Excel.

op_file_uf_1 = op_file_unformatted[["order_id_x","product_id"]]

#op_file_uf_1.to_csv('previous_orders.csv')
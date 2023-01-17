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
order_prod_prior = pd.read_csv("../input/order_products__prior.csv")

order_prod_prior.head()
orders = pd.read_csv("../input/orders.csv")

orders.head()
sample_sub = pd.read_csv("../input/sample_submission.csv")

sample_sub.head()
order_prod_train = pd.read_csv("../input/order_products__train.csv")

order_prod_train.head()
joinnone = order_prod_prior[['order_id','product_id']].merge(orders[['order_id', 'user_id']], on='order_id').groupby(['user_id','product_id']).size()

joinnone.head()
asd = pd.DataFrame(joinnone)

asd.reset_index(inplace=True)

asd.head()
asd['values'] = asd[['product_id',0]].apply(tuple, axis=1)

asd = asd[['user_id', 'values']]

asd.head()
def products_concat(vet):

    out = ''

    

    #vet is a pd.Series

    for prod in [x[0] for x in sorted(vet, key = lambda x: x[1], reverse = True) if x[1] > 1]:

        if prod > 0:

            out += str(int(prod)) + ' '

    

    if out != '':

        return out.rstrip()

    else:

        return 'None'



ilmodel2 = pd.DataFrame(asd.groupby(['user_id']).values.apply(products_concat))
ilmodel2.head()
AAB = orders[orders['eval_set'] == 'test'][['order_id', 'user_id']].set_index("user_id").merge(ilmodel2, right_index = True, left_index = True )

AAB.to_csv("hello3_csv.csv")
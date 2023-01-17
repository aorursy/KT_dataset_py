import pandas as pd

import numpy as np

import operator



# special thanks to Nick Sarris who has written a similar notebook

# reading data

#mdf = 'c:/Users/John/Documents/Research/entropy/python/InstaCart/data/'

mdf = '../input/'

print('loading prior orders')

prior_orders = pd.read_csv(mdf + 'order_products__prior.csv', dtype={

        'order_id': np.int32,

        'product_id': np.int32,

        'add_to_cart_order': np.int16,

        'reordered': np.int8})

print('loading orders')

orders = pd.read_csv(mdf + 'orders.csv', dtype={

        'order_id': np.int32,

        'user_id': np.int32,

        'eval_set': 'category',

        'order_number': np.int16,

        'order_dow': np.int8,

        'order_hour_of_day': np.int8,

        'days_since_prior_order': np.float32})



pd.set_option('display.float_format', lambda x: '%.3f' % x)

# removing all user_ids not in the test set from both files to save memory

# the test users present ample data to make models. (and saves space)

test  = orders[orders['eval_set'] == 'test' ]

user_ids = test['user_id'].values

orders = orders[orders['user_id'].isin(user_ids)]

order_ids = orders['order_id'].values

prior_orders = prior_orders[prior_orders['order_id'].isin(order_ids)]

#del test

test.shape


# Calculate the Prior : p(reordered|product_id)

prior = pd.DataFrame(prior_orders.groupby('product_id')['reordered'].agg([('number_of_orders',len),

        ('sum_of_reorders','sum')]))

prior['prior_p'] = (prior['sum_of_reorders']+1)/(prior['number_of_orders']+2) # Informed Prior

#prior['prior_p'] = 1/2  # Flat Prior

prior.drop(['number_of_orders','sum_of_reorders'], axis=1, inplace=True)

print('Here is The Prior: our first guess of how probable it is that a product be reordered once it has been ordered.')



prior.head(3)
# merge everything into one dataframe and save any memory space



comb = pd.DataFrame()

comb = pd.merge(prior_orders, orders, on='order_id', how='right')

# slim down comb - 

comb.drop(['order_dow','order_hour_of_day'], axis=1, inplace=True)

del prior_orders

del orders

prior.reset_index(inplace = True)

comb = pd.merge(comb, prior, on ='product_id', how = 'left')

print('combined data in DataFrame comb')

comb.head(3)


user = pd.DataFrame(columns =('order_id', 'products'))

z = pd.DataFrame()

prods = pd.DataFrame()

ords = pd.DataFrame()

n = 0



for user_id in user_ids:

    exp_reorders = 0

    z = comb[comb.user_id == user_id]

    prods = z.groupby(['product_id'])['reordered'].agg({"m": np.sum})

    prods.loc[:,'m'] = prods.loc[:,'m'] + 1

    prods.loc[:,'tot_ords'] = max(comb.order_number[comb.user_id == user_id]) - 1

    prods.loc[:,'prior'] = z.groupby(['product_id'])['prior_p'].agg(np.mean)

    prods.loc[:,'prob'] = (prods.loc[:,'m'] + 1)/(prods.loc[:,'tot_ords'] + 2)

    prods.loc[:,'post'] = (prods.loc[:,'tot_ords'] * prods.loc[:,'prob'] \

                       + prods.loc[:,'prior'] * 5.)/(prods.loc[:,'tot_ords'] + 5.)

    prods = prods.sort_values('post', ascending=False).reset_index()



    ords = z.groupby(['order_number'])['reordered'].agg({"n": np.sum})

    last_o_id = max(z.order_id[z.eval_set == 'test'])

    if len(ords) == 4:

        exp_reorders = round((ords.n.iloc[-2] + ords.n.iloc[-3])/2.,0)

    elif len(ords) == 5:

        exp_reorders = round((ords.n.iloc[-2] + ords.n.iloc[-3] + ords.n.iloc[-4])/3.,0)

    else:

        exp_reorders = round((ords.n.iloc[-2] + ords.n.iloc[-3] + ords.n.iloc[-4] \

                    + ords.n.iloc[-5])/4.,0)

    if exp_reorders != 0:

        prod_str = ""

        for i in range(int(exp_reorders)):

            prod_str = prod_str + " " + str(int(prods.iloc[i,0]))

        s = [[int(last_o_id), prod_str]]

        user = user.append(pd.DataFrame(s, columns = ['order_id', 'products']))

        n = n + 1

    else:

        s = [[int(last_o_id), "None"]]

        user = user.append(pd.DataFrame(s, columns = ['order_id', 'products']))

        n = n + 1

user[['order_id', 'products']].to_csv(mdf + 'bayesian-flat.csv', index=False)

user.sort_values('order_id').head(5)
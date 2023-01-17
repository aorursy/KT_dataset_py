from IPython.display import HTML

HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/3HJtmx5f1Fc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import os

import pandas as pd

import numpy as np
with open('../input/hashcode-drone-delivery/busy_day.in') as file:

    line_list = file.read().splitlines()
len(line_list)
line_list[:4]
first_section = line_list[0].split()

first_section
second_section = line_list[1].split()

second_section
third_section = line_list[2].split()

len(third_section)
third_section[:10]
fourth_section = line_list[3].split()

fourth_section
fifth_section = line_list[4].split()

fifth_section
sixth_section = line_list[5].split()

len(sixth_section)
seventh_section = line_list[6].split()

seventh_section
eightth_section = line_list[7].split()

len(eightth_section)
order_section = line_list[24].split()

order_section
first_order_delivered = line_list[25].split()

first_order_delivered
first_order_items = line_list[26].split()

first_order_items
len(line_list)
# Products

weights = line_list[2].split()

products_df = pd.DataFrame({'weight': weights})



wh_count = int(line_list[3])

wh_endline = (wh_count*2)+4



wh_invs = line_list[5:wh_endline+1:2]

for i, wh_inv in enumerate(wh_invs):

    products_df[f'wh{i}_inv'] = wh_inv.split()



products_df = products_df.astype(int)



# Warehouses

wh_locs = line_list[4:wh_endline:2]

wh_rows = [wl.split()[0] for wl in wh_locs]

wh_cols = [wl.split()[1] for wl in wh_locs]



warehouse_df = pd.DataFrame({'wh_row': wh_rows, 'wh_col': wh_cols}).astype(np.uint16)



# Orders

order_locs = line_list[wh_endline+1::3]

o_rows = [ol.split()[0] for ol in order_locs]

o_cols = [ol.split()[1] for ol in order_locs]



orders_df = pd.DataFrame({'row': o_rows, 'col': o_cols})



orders_df[orders_df.duplicated(keep=False)].sort_values('row')



orders_df['product_count'] = line_list[wh_endline+2::3]



order_array = np.zeros((len(orders_df), len(products_df)), dtype=np.uint16)

orders = line_list[wh_endline+3::3]

for i,ord in enumerate(orders):

    products = [int(prod) for prod in ord.split()]

    order_array[i, products] = 1



df = pd.DataFrame(data=order_array, columns=['p_'+ str(i) for i in range(400)], 

                    index=orders_df.index)



orders_df = orders_df.astype(np.uint16).join(df)
products_df.head()
warehouse_df.head()
orders_df.head()
len(orders_df)
import pandas_profiling as pdp
profile_products_df = pdp.ProfileReport(products_df)
profile_products_df
profile_warehouse_df = pdp.ProfileReport(warehouse_df)
profile_warehouse_df
profile_orders_df = pdp.ProfileReport(orders_df)
profile_orders_df
first_line = '10'

first_line
second_line = ['0 L 1 2 3']

second_line
third_line = ['0 D 0 0 1']

third_line
fourth_line = ['1 L 1 2 1']

fourth_line
fifth_line = ['1 D 2 2 1']

fifth_line
submission = ''
submission += ''.join(first_line)

submission += '\n'

submission += ' '.join(second_line)

submission += '\n'

submission += ' '.join(third_line)

submission += '\n'

submission += ' '.join(fourth_line)

submission += '\n'

submission += ' '.join(fifth_line)

submission += '\n'
submission
sub_file = open("submission.csv", "w")

sub_file.write(submission)

sub_file.close()
sub = pd.read_csv('submission.csv')

sub.head()
submission = '2\n'

submission += '0 L 6 163 1\n'

submission += '0 D 1 163 1\n'



sub_file = open("submission.csv", "w")

sub_file.write(submission)

sub_file.close()



sub = pd.read_csv('submission.csv')

sub.head()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
with open('../input/hashcode-drone-delivery/busy_day.in') as file:

    data_list = file.read().splitlines()
print('rows of grid,columns of grid,drones,turns, maxpay load in units(u):',data_list[0], 

      '\n Different product types:',data_list[1],

     '\n product types weigh:',data_list[2],

      '\n warehouses:',data_list[3],

      '\n First warehouse location at first warehouse (row, column):',data_list[4],

      '\n Inventory of products:',data_list[5],

      '\n second warehouse location (row, column)  :',data_list[6],

      '\n Inventory of products at second ware house:',data_list[7],

      '\n Number of orders:',data_list[24],

      '\n First order to be delivery at:',data_list[25],

      '\n Number of items in order:',data_list[26],

      '\n Items of product types:',data_list[27]    )   
# lets get all the 10 ware house co-ordinates

ware_house_locs = data_list[4:24:2]

ware_house_rows = [ware_house_r.split()[0] for ware_house_r in ware_house_locs]

ware_house_cols = [ware_house_c.split()[1] for ware_house_c in ware_house_locs]



warehouse_df = pd.DataFrame({'ware_house_row': ware_house_rows, 'ware_house_col': ware_house_cols}).astype(np.uint16)

warehouse_df
# Lets aggregate all the products available at their respoective ware houses



cols=[f'ware_house_{i}' for i in range(len(warehouse_df))]



products_df = pd.DataFrame([x.split() for x in data_list[5:24:2]]).T



products_df.columns=cols
products_df
# lets add weight of each product to product_df



products_df['prod_weight']= data_list[2].split()
products_df=products_df.astype('int')

products_df
cols_order=[f'prod_{i}' for i in range(19)]



order_df = pd.DataFrame([x.split() for x in data_list[27:3775:3]]).fillna(0).astype('int')



order_df.columns=cols_order



order_df['order_items'] = data_list[26:3775:3]



order_df['order_coor_x'] = [x.split()[0] for x in data_list[25:3775:3]]

order_df['order_coor_y'] = [x.split()[1] for x in data_list[25:3775:3]]



order_df=order_df.astype('int')



order_df
order_df.dtypes
import seaborn as sns

sns.scatterplot(data=order_df, x="order_coor_x", y="order_coor_y")#, hue="time")
#Distribution of the warehouses

sns.scatterplot(data=warehouse_df, x="ware_house_row", y="ware_house_col")
import matplotlib.pyplot as plt



x = range(400)

y = range(400,600)

fig = plt.figure(figsize=(12,10))

ax1 = fig.add_subplot(111)



ax1.scatter(warehouse_df['ware_house_row'], warehouse_df['ware_house_col'], s=50, c='b', marker="o", label='ware_house')

ax1.scatter(order_df['order_coor_x'], order_df['order_coor_y'], s=20, c='r', marker="x", label='orders')

plt.legend(loc='upper left');

plt.show()
# Frequency of orders

x= [len(x.split()) for x in data_list[27:3775:3]]

x = pd.DataFrame(x, columns=["orders"])

ax = sns.distplot(x)
fig,ax = plt.subplots(1,1)



ax.hist(x.orders, [x for x in range(0,19)])



plt.show()
sns.barplot(x="orders", y=x.orders.value_counts(),data=x)
inventory_array = np.zeros((400, 600, 400), dtype=np.uint16)



wh = warehouse_df.to_numpy()

inv = products_df.drop(columns=["prod_weight"]).T.to_numpy()

inventory_array[wh[:, 0], wh[:, 1]] = inv



inventory_array.sum()
print(inventory_array[182,193,1], 

    np.array_equal(inventory_array.sum(axis=(0, 1)), inv.sum(axis=0)))
import pandas_profiling as pdp

profile_products_df = pdp.ProfileReport(products_df)
profile_products_df
profile_warehouse_df = pdp.ProfileReport(warehouse_df)
profile_warehouse_df
import pandas as pd

profile_order_df = pdp.ProfileReport(order_df)
profile_order_df
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
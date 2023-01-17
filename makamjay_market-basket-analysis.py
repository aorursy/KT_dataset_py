import pandas as pd

import numpy as np
data = pd.read_csv('C:/Users/makam/Desktop/Data Analytics/superstore_dataset2011-2015.csv',encoding='unicode_escape')
data
data1 =pd.DataFrame()

order1 = pd.Series(data['Order ID'])

product_name = pd.Series(data['Product Name'])

data1['Order ID']=order1

data1['Product Name']=product_name
data1
cont = pd.get_dummies(data1['Product Name'],prefix='',drop_first=False)

#Adding the results to the master dataframe

data2 = pd.concat([data1,cont],axis=1)
del data2['Product Name']
data2
len(data1['Product Name'].unique())
len(data2.columns)
col = list(data2.columns)

del col[0]
data2
d = data2.groupby(data2['Order ID']).sum()  
d
d.to_csv('C:/Users/makam/Desktop/Data Analytics/Project/ovspmat.csv')
orders = pd.Series(d.index)

orders
len(orders)
d
#Here we will consider only those orders where the basket size is greater than 2. 

#This is done to reduce computational load.

d_t = d.T
d_t
f=[]

for col in orders:

    if d_t[col].sum()>1:

        f.append(col)
f
len(f)
t=pd.DataFrame()
t['Order ID']=f
datafinal = t.merge(d,on='Order ID',how='inner')
datafinal.shape
orders = list(datafinal.columns)

del orders[0]

orders
#Here we will only consider those items where the count of its sale is greater than 20

product_support_dict = []

for column in orders:

    if sum(datafinal[column])<=20:

        product_support_dict.append(column)
len(product_support_dict)
for col in product_support_dict:

    del datafinal[col]
datafinal.shape
datafinal
orders = list(datafinal['Order ID'])

products = list(datafinal.columns.values)

products
del products[0]
datafinal.set_index('Order ID',inplace=True)

d_t=datafinal.T
f=[]

for col in list(d_t.columns):

    if d_t[col].sum()>1:

        f.append(col)
len(f)
t=pd.DataFrame()

t['Order ID']=f

datafinal = t.merge(datafinal,on='Order ID',how='inner')
datafinal.shape
datafinal.to_csv('C:/Users/makam/Desktop/Data Analytics/Project/Final Project/finalmatrix.csv')
datafinal.set_index('Order ID',inplace=True)
orders = list(datafinal.index.values)

products = list(datafinal.columns.values)
transaction_matrix = datafinal.as_matrix()

# get number of rows and columns

rows, columns = transaction_matrix.shape

# init new matrix

frequent_items_matrix = np.zeros((datafinal.shape[1],datafinal.shape[1]))

# compare every product with every other

for this_column in range(0, columns):

    for next_column in range(this_column + 1, columns):

        # multiply product pair vectors

        product_vector = transaction_matrix[:,this_column] * transaction_matrix[:,next_column]

        # check the number of pair occurrences in baskets

        count_matches = sum((product_vector)>0)

        # save values to new matrix

        frequent_items_matrix[this_column,next_column] = count_matches
import matplotlib.pyplot as plt

plt.imshow(frequent_items_matrix)

plt.colorbar()

plt.show()
# and finally combine product names with data

frequent_items_df = pd.DataFrame(frequent_items_matrix, columns = datafinal.columns.values, index = datafinal.columns.values)

 

import seaborn as sns

# and plot

sns.heatmap(frequent_items_df)
%matplotlib inline

sns.heatmap(frequent_items_df)
# extract product pairs with minimum frequency(treshold) basket occurrences

from collections import OrderedDict 

def extract_pairs(treshold,frequent_items_matrix,product_names):

    output = {}

    # select indexes with larger or equal n

    matrix_coord_list = np.where(frequent_items_matrix >= treshold)

    # take values

    row_coords = matrix_coord_list[0]

    column_coords = matrix_coord_list[1]

    # generate pairs

    for index, value in enumerate(row_coords):

        #print index

        row = row_coords[index]

        column = column_coords[index]

        # get product names

        first_product = product_names[row]

        second_product = product_names[column]

        # number of basket matches

        matches = frequent_items_matrix[row,column]

        # put key values into dict

        output[first_product+"-"+second_product] = matches

 

    # return sorted dict

    sorted_output = OrderedDict(sorted(output.items(), key=lambda x: x[1]))

    return sorted_output

 

# plot pairs with minimum frequency of 1 basket matches

pd.Series(extract_pairs(1,frequent_items_matrix,products)).plot(kind="barh")

finallist = pd.Series(extract_pairs(1,frequent_items_matrix,products))
finallist = pd.Series(extract_pairs(1,frequent_items_matrix,products))
finallist 
len(finallist)
finallistD = pd.DataFrame()

finallistD['Count']=finallist
#This list is the pair of all the items with frequency of coming together in a basket.

finallistD
finallistD.to_csv('C:/Users/makam/Desktop/Data Analytics/Project/Final Project/FinalPairs3.csv')
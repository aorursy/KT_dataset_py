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


dep_set = pd.read_csv("../input/departments.csv")

aisles_set = pd.read_csv("../input/aisles.csv")

order_product_train_set = pd.read_csv("../input/order_products__train.csv")

product_set = pd.read_csv("../input/products.csv")

order_set = pd.read_csv("../input/orders.csv")

order_product_prior_set = pd.read_csv("../input/order_products__prior.csv")
#Merge order product prio with product set (to see who's the product)

merged1 = pd.merge(order_product_prior_set, product_set, on='product_id')



#group it by product and reorders

test1 = merged1.groupby(["product_id", "reordered"]).size().reset_index(name="Count")

#test1 contains all merged1 data that was grouped by product id and reordered



#sort them for count descending 

sortedTest1 = test1.sort_values(by='Count', ascending=False)



#

sortedTest1.head(5)
#Function that returns the productId that contains the max Count of reorders.

def getProductId(test1Array, maxCount):

    for i in range(1,len(test1Array)-1):

        if test1Array[i][2] == maxCount and test1Array[i][1] == 1:

            print('Max count found!')

            print('Product ID: %d' % test1Array[i][0])

            print('Reorder: %d' % test1Array[i][1])

            print('Count: %d ' % test1Array[i][2])

            return test1Array[i][0]

    return -1    





#dataframe in array form

test1Array = test1.values



maxCount = test1['Count'].max()

#test1Array[0][2] <- get certain value!

#test1Array[x][y]

#x = the row in the dataframe

#y = the category

###y = 0 -> product ID

###y = 1 -> Reorder (1 or 0)

##y = 2 -> Count









print('Max count from the dataframe: %d' % maxCount)

#productId of the max count of reorders

productId = getProductId(test1Array, maxCount)

print('kudasai')

print('The productId of the maxCount: %d' % productId)



















#slimOrder contains the count of orders on each day of the week.

#0 - Sunday

#1 - Monday

#2 - Tuesday

#3 - Wednesday

#4 - Thursday

#5 - Friday

#6 - Saturday

slimOrder = order_set.groupby(['order_dow']).size().reset_index(name="count")

slimOrder

#product_set

#aisles_set



mergeSet = pd.merge(product_set, aisles_set, on="aisle_id")



mergeSet = mergeSet[["product_id", "product_name", "aisle"]]

mergeSet



#order_product_train_set

#order_set



combine = pd.merge(order_product_train_set, order_set, on="order_id")

combine[["order_id", "product_id", "order_dow"]]
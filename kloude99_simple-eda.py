import pandas as pd

import numpy as np

import seaborn as sns

import os

sns.set(color_codes = True)

%matplotlib inline
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Aisles file

aisles = pd.read_csv("../input/aisles.csv")

print("Total Aisles: {}".format(aisles.shape[0]))

aisles.head()
#Departments file

depts = pd.read_csv("../input/departments.csv")

print("Total Departments: {}".format(depts.shape[0]))

depts.head()
#Order file

orders = pd.read_csv("../input/orders.csv")

print("Total Orders: {}".format(orders.shape[0]))

orders.head()
#Products file

products = pd.read_csv('../input/products.csv')

print("Total Number of Products: {}".format(products.shape[0]))

products.head()
#Order_Products_train file

order_products_train = pd.read_csv('../input/order_products__train.csv')

order_products_train.head()
#Order_Products_Prior file

order_products_prior = pd.read_csv('../input/order_products__prior.csv')

order_products_prior.head(10)
#Join Prodcuts, Aisles and Departments to create product major

product_major = pd.merge(products,aisles,how="left", on = "aisle_id")

product_major= pd.merge(product_major, depts, how ="left", on = "department_id")

product_major.head(20)
#EDA on products

#Department-product distibution

sns.countplot(y = "department", 

              data = product_major, 

              palette= "Greens_d", 

              orient= "h",

              order=product_major.department.value_counts().index

             )



#A lot of products in snacks, personal care, pantry, beverages, frozen, dairy eggs department 
#Aisle- product distribution

sns.countplot(y = "aisle",

              data = product_major,

              palette= "Blues_d",

              orient= "h",

              order=product_major.aisle.value_counts().iloc[:35].index

             )

#Missing aisle data

#Highest products in candy chocolate aisle because of their size and impulse buying of the related products so giving more choice is the key here

#Same with ice-cream. So the first two category of aisles are mostly unplanned purchases(depending on the product segment)
product_major.aisle.value_counts()['missing']



#Find out missing values

missing = np.where(product_major.aisle=='missing')

#print(missing[0][:])



#All values for aisle_id = 100 are missing

print("Missing Aisle ID: {}".format(product_major.aisle_id.loc[missing[0]].unique())) 



#All values for department_id = 21 are missing

print("Missing Department ID: {}".format(product_major.department_id.loc[missing[0]].unique()))

print("\n")



#Verify in the relevant tables

print(aisles.loc[np.where(aisles.aisle_id==100)])



print(depts.loc[np.where(depts.department_id==21)])
#Merging Order_product_prior and product_major

ord_prd_major = pd.merge(order_products_prior,product_major,on="product_id",how ="left")
ord_prd_major.head(20)
dep_prob = []

for dep in ord_prd_major.department.unique():

    dep_prob = np.append(dep_prob, 100*ord_prd_major.department.value_counts()[dep]/ord_prd_major.shape[0])
#Probability of choosing a particular department 

dep_prob_df = pd.DataFrame(ord_prd_major.department.unique())

dep_prob_df['Probabilities'] = dep_prob
dep_prob_df
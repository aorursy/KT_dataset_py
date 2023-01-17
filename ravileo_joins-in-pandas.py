import pandas as pd
product=pd.DataFrame({

    'Product_ID':[101,102,103,104,105,106,107],

    'Product_name':['Watch','Bag','Shoes','Smartphone','Books','Oil','Laptop'],

    'Category':['Fashion','Fashion','Fashion','Electronics','Study','Grocery','Electronics'],

    'Price':[299.0,1350.50,2999.0,14999.0,145.0,110.0,79999.0],

    'Seller_City':['Delhi','Mumbai','Chennai','Kolkata','Delhi','Chennai','Bengalore']

})
product
customer=pd.DataFrame({

    'id':[1,2,3,4,5,6,7,8,9],

    'name':['Olivia','Aditya','Cory','Isabell','Dominic','Tyler','Samuel','Daniel','Jeremy'],

    'age':[20,25,15,10,30,65,35,18,23],

    'Product_ID':[101,0,106,0,103,104,0,0,107],

    'Purchased_Product':['Watch','NA','Oil','NA','Shoes','Smartphone','NA','NA','Laptop'],

    'City':['Mumbai','Delhi','Bangalore','Chennai','Chennai','Delhi','Kolkata','Delhi','Mumbai']

})
customer
pd.merge(product,customer,on='Product_ID')
# if the column names are different

pd.merge(product,customer,left_on='Product_name',right_on='Purchased_Product')
## seller and customer both belong to the same city.

pd.merge(product,customer,how='inner',left_on=['Product_ID','Seller_City'],right_on = ['Product_ID','City'])
pd.merge(product,customer,on='Product_ID',how='outer')
pd.merge(product,customer,on='Product_ID',how='outer',indicator=True)
pd.merge(product,customer,on='Product_ID',how='left')
pd.merge(product,customer,on='Product_ID',how='right')
# Dummy dataframe with duplicate values



product_dup=pd.DataFrame({'Product_ID':[101,102,103,104,105,106,107,103,107],

                          'Product_name':['Watch','Bag','Shoes','Smartphone','Books','Oil','Laptop','Shoes','Laptop'],

                          'Category':['Fashion','Fashion','Fashion','Electronics','Study','Grocery','Electronics','Fashion','Electronics'],

                          'Price':[299.0,1350.50,2999.0,14999.0,145.0,110.0,79999.0,2999.0,79999.0],

                          'Seller_City':['Delhi','Mumbai','Chennai','Kolkata','Delhi','Chennai','Bengalore','Chennai','Bengalore']})
product_dup
pd.merge(product_dup,customer,on='Product_ID',how='inner')
# Drop duplicates

pd.merge(product_dup.drop_duplicates(),customer,on='Product_ID',how='inner')
# "Validate" to keep the duplicates

pd.merge(product_dup,customer,on="Product_ID",how='inner',validate='many_to_many')
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
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
with open("../input/hashcode-drone-delivery/busy_day.in") as file: 
     data   = file.read().splitlines()
data
lis=data[0] 
lis=lis.split()

print(" "*60+"Data Analysis")
print("="*200)
print("Number of rows : "+str(lis[0])) 
print("Number of columns : "+str(lis[1])) 
print("Number of drones : "+str(lis[2])) 
print("Number of turns : "+str(lis[3])) 
print("Maximum payload Weight that drone can carry : "+str(lis[4]))  
print("Number of products that has be to be delivered : "+str(data[1])) 
print("let me know what and all product types and with respective weights ")  
print((" "*60+"Product Weights"))
print("="*100) 
print("Product weights : "+str(data[2])) 


len(data[2])
len(data[13])
print(" "*60+"Information about Ware houses ") 
print("="*100) 
print("Number of ware houses : "+str(data[3])) 
print("ware house locations and what and all inventory products that belong to each ware house ") 
print("=="*100) 
print("First Ware house location (lat,long) : "+str(data[4])) 
print("Inventory Products : "+str(data[5])) 
print("Second Ware house location (lat,long) : "+str(data[6])) 
print("Inventory products : "+str(data[7])) 
print("Third Ware house location (lat,long) : "+str(data[8])) 
print("Inventory products : "+str(data[9])) 
print("fourth Ware house location (lat,long) : "+str(data[10])) 
print("Inventory Products : "+str(data[11])) 
print("Fifth Ware house locatinon(lat,long) : "+str(data[12])) 
print("Inventory Products : "+str(data[13])) 
print("Sixth Ware house loocation : "+str(data[14])) 
print("Inventory Products : "+str(data[15])) 
print("Seventh Ware house location : "+str(data[16])) 
print("Inventory Products : "+str(data[17])) 
print("eight Ware house location : "+str(data[18])) 
print("Inventory Products : "+str(data[19])) 
print("ninth Ware house location : "+str(data[20])) 
print("Inventory Products : "+str(data[21])) 
print("tenth Ware house location : "+str(data[22])) 
print("Inventory Products : "+str(data[23]))
print("Number of orders : "+data[24]) 
print("=="*40)  
print("First order") 
print("==="*6)
print("Delivered location : "+data[25]) 
print("Denote number of product items : "+data[26]) 
print("Denote number of product types : "+data[27]) 
print("second order ") 
print("=="*30) 
print("delivered location : "+data[28]) 
print("Denotes number of product items :"+str(data[29])) 
print("Denotes number of types : "+str(data[30])) 
print("Third order") 
print("=="*60) 
print("Delivered location : "+str(data[31])) 
print("Denotes number of product items : "+str(data[32])) 
print("Denotes number of types : "+str(data[33])) 
print("=="*500)
print("It has continued  untill it reaches the total number of orders or completion of order that we are assigning to a particular drone")
len(data[5])
# lets get all the 10 ware house co-ordinates
ware_house_locs = data[4:24:2]
ware_house_rows = [ware_house_r.split()[0] for ware_house_r in ware_house_locs]
ware_house_cols = [ware_house_c.split()[1] for ware_house_c in ware_house_locs]

warehouse_df = pd.DataFrame({'Latitude': ware_house_rows, 'Longitude': ware_house_cols}).astype(np.uint16)
warehouse_df
plt.figure(figsize=(12,8))
plt.scatter(x="Latitude",y="Longitude",data=warehouse_df,marker="*",s=500,c="darkgreen")
plt.xlabel("Latitude") 
plt.ylabel("Longitude") 
plt.title("Warehouse Locations")
warehouse_inventory=data[5:25:2] 
products=[inven.split() for inven in warehouse_inventory] 
warehouses=["warehouse "+str(i) for i in range(10)] 
df=pd.DataFrame(products).T 
df.columns=warehouses 
df
df["product_weight"]=data[2].split()
df
# We have to deliver the products based on the customer address and the delivered locations are in between 400 and 600 grid 
delivered_location=data[25:3775:3] 
delivered_row=[delivered.split()[0] for delivered in delivered_location]
deliverd_column=[delivered1.split()[1] for delivered1 in delivered_location] 
delivered_location=pd.DataFrame({'delivered_row':delivered_row,"delivered_col":deliverd_column}) 
delivered_location
cols_order=[f'prod_{i}' for i in range(19)]
orders= pd.DataFrame([x.split() for x in data[27:3775:3]]).fillna(0).astype('int')
orders.columns=cols_order
orders['order_items'] = data[26:3775:3] 
orders["delivered_location_row"]=delivered_location["delivered_row"] 
orders["delivered_location_col"]=delivered_location["delivered_col"]  
orders
plt.figure(figsize=(20,8))
plt.scatter(x=orders["delivered_location_row"],y=orders["delivered_location_col"])  
fig=plt.figure(figsize=(20,10)) 
ax1=fig.add_subplot(111)  
x=range(400) 
y=range(400,600)
ax1.scatter(x="Latitude",y="Longitude",data=warehouse_df,marker="*",s=1000,c="purple") 
ax1.scatter(x="delivered_row",y="delivered_col",data=delivered_location,s=50)
#Total number of products when compared to each and every respective ware house 
products=[] 
for i in df.columns: 
    products.append(df[i].sum())
# If We can observe clearly in the given dataset most of the product types are 400 
print("Number of Product types : "+str(data[1]))
#But they mentioned the condition is Product weight shoulld not be greater than 200 kg in each drone  
#Let's try to check for each product weight  
#It's Perfect product weights are normally distributed 
plt.figure(figsize=(12,10))
sns.distplot(df["product_weight"],color="green")
#As we can see order items are almost normal distributed
plt.figure(figsize=(12,10))
sns.distplot(orders["order_items"],color="purple")

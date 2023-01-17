import pandas as pd
import os
df= pd.read_csv("../input/sales-analysis-panda/Sales_April_2019.csv")

files=[file for file in os.listdir("../input/sales-analysis-panda")]

all_data=pd.DataFrame()

for file in files:
    df=pd.read_csv("../input/sales-analysis-panda/"+file)
    all_data=pd.concat([all_data,df])

all_data.to_csv("all_data.csv",index=False)
all_data=pd.read_csv("all_data.csv")
all_data.head()
all_data=all_data.dropna(how='all')
allna=all_data[all_data.isna().any(axis=1)]
allna.shape
tempdf=all_data[all_data['Order Date'].str[0:2] == 'Or']
print(tempdf.shape)
print(all_data.shape)
tempdf.head()
all_data=all_data[all_data['Order Date'].str[0:2] != 'Or']
print(all_data.shape)
all_data.head()
all_data['Quantity Ordered']=all_data["Quantity Ordered"].astype("int32")
all_data['Price Each']=all_data["Price Each"].astype("float64")
all_data['Month']=all_data['Order Date'].str[0:2]


all_data['Month']=all_data['Month'].astype('int32')
all_data.head()
all_data['Sales']=all_data['Quantity Ordered']*all_data['Price Each']
all_data.head()
all_data["City"]=all_data["Purchase Address"].apply(lambda x : x.split(',')[1]+' '+ x.split(',')[2].split(' ')[1])
all_data.head()
all_data.groupby('Month').sum().sort_values('Sales',ascending=False)
all_data.groupby('Month').max()
import matplotlib.pyplot as plt

highestsales_month=all_data.groupby('Month').sum()

months=range(1,13)

plt.bar(months,highestsales_month['Sales'])
plt.title('Total Sales per Month')
plt.xlabel('Month')
plt.ylabel('Sales Value')

plt.show()
highestsale_city=all_data.groupby('City').sum()
highestsale_city.head(10)

all_data['City'].shape
cities=[city for city,df in all_data.groupby('City')]

plt.bar(cities,highestsale_city['Sales'])
plt.xticks(cities,rotation='vertical',size=8)
plt.xlabel('City Name')
plt.title('City vs Sales')
plt.ylabel('Sales')
plt.show()
all_data['Order Date']=pd.to_datetime(all_data['Order Date'])
all_data['Hour']=all_data['Order Date'].dt.hour
all_data['Minute']=all_data['Order Date'].dt.minute
all_data.head(10)
hours=[hour for hour,df in all_data.groupby('Hour')]
minutes=[minute for minute,df in all_data.groupby('Minute')]

plt.plot(hours,all_data.groupby(['Hour']).count())
plt.xticks(hours,size=8)
plt.xlabel('Hour')
plt.ylabel('No of Purchases')
plt.grid()
plt.show()
df=all_data[all_data['Order ID'].duplicated(keep=False)]
df['Grouped']=df.groupby('Order ID')['Product'].transform(lambda x: ', '.join(x))
df=df[["Order ID","Grouped"]].drop_duplicates()
df.head()


from itertools import combinations
from collections import Counter

count =Counter()

for row in df['Grouped']:
    row_list=row.split(',')
    count.update(Counter(combinations(row_list,2)))
    
for key,values in count.most_common(10):
    print(key,values)
    
product_group=all_data.groupby('Product')
quantity_total= product_group['Quantity Ordered'].sum()

product=[product for product,df in product_group]

plt.bar(product,quantity_total)
plt.xticks(product,rotation='vertical',size=8)
plt.xlabel('Product',size=12)
plt.ylabel('Quantity Sold',size=12)
plt.title('Quantity of Products sold',size=15.0)
plt.show()

prices=all_data.groupby('Product').mean()['Price Each']

fig,ax1= plt.subplots()
ax2=ax1.twinx()
ax1.bar(product,quantity_total,color='g')
ax2.plot(product,prices,'b-')

ax1.set_xlabel('Products')
ax1.set_xticklabels(product,rotation='vertical')
ax1.set_ylabel('Quantity',color='g')
ax2.set_ylabel('Price',color='b')
plt.show()

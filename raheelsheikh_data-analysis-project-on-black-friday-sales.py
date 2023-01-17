import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#read the file

retaildata = pd.read_csv("../input/BlackFriday.csv")
print(retaildata.head())
#make a copy of data
retaildata_c = pd.DataFrame(retaildata)
sns.set(style= "darkgrid")
sns.catplot(x="Age", kind="count",data=retaildata_c);
plt.title('Highest Purchasing Age Group')
unique_gender = retaildata_c.Gender.unique()

countF = retaildata_c[retaildata_c['Gender'] == 'F'].count() 
countM = retaildata_c[retaildata_c['Gender'] == 'M'].count() 

values= [countF.Gender,countM.Gender]
labels = ['Female', 'Male']
explode = (0.2, 0)
plt.pie(values, labels= values,explode=explode,autopct='%1.1f%%',counterclock=False, shadow=True)
plt.title('Ratio of Purchases made by Gender')
plt.legend(labels,loc=3)
plt.show()
sns.set(style= "darkgrid")
sns.catplot(x="City_Category", kind="count",data=retaildata_c);
plt.title('Total Sales made City wise')
unique_Product_Cat_1 = retaildata_c.Product_Category_1.unique()
print(unique_Product_Cat_1)
list_product_cat_1 = []
list_product_count = []
for prod in unique_Product_Cat_1:
    prodname = prod
    count = retaildata_c[retaildata_c['Product_Category_1'] == prodname].count()
    list_product_cat_1.append(prodname)
    list_product_count.append(count.Product_Category_1)
    
print(list_product_cat_1,list_product_count)
    
#explode = (0.2, 0)
plt.pie(list_product_count, labels= list_product_cat_1,autopct='%1.1f%%',radius=2,pctdistance=1.3,labeldistance=1.1,counterclock=False, shadow=True)
plt.title('Purchases by category')
plt.legend(list_product_cat_1,loc=3)
plt.show()
productwise_revenue = retaildata_c.groupby(['Product_Category_1'])['Purchase'].sum().reset_index()

plt.figure(figsize=(15,8))
plt.ticklabel_format(style='plain', axis='y',useOffset=False)
plt.title('Product category that generated Maximum revenue')
ax = sns.barplot(x='Product_Category_1', y='Purchase',data=productwise_revenue)
Citywise_prod_purchase = retaildata_c.groupby(['Product_Category_1','City_Category']).count().reset_index()
Citywise_prod_purchase[['Product_Category_1','City_Category','Purchase']]
Citywise_prod_purchase = pd.DataFrame(Citywise_prod_purchase)
Citywise_prod_purchase['Product_Category_1'] = Citywise_prod_purchase.Product_Category_1.astype(str)
Citywise_prod_purchase['Product_by_city'] = Citywise_prod_purchase[['Product_Category_1', 'City_Category']].apply(lambda x: ''.join(x), axis=1)
print(Citywise_prod_purchase[['Product_by_city','Purchase']].head())

 
plt.figure(figsize=(25,15))
plt.plot( 'Product_by_city', 'Purchase', data=Citywise_prod_purchase, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4) 
plt.title('City wise Purchase of Product')
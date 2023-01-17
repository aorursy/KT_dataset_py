import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
df = pd.read_pickle('../input/IowaCleaned')
df.head()
df.groupby('item_number')[['bottles_sold']].sum()
#count how many items in total data set - 7392
df.groupby('vendor_number')[['bottles_sold']].sum()
#count how many vendors - 271
vbs = df.groupby('vendor_name')[['bottles_sold']].sum()
#vendor that sold most bottles
vbs = vbs.sort_values('bottles_sold',ascending=False)
#sort by top down
vbs = vbs.reset_index()
#reset the index so columns are named correctly
vbs20 = vbs.head(n=20)
#take the top 20 vendors
vbs20.columns = ['Vendor Name','Total Bottles Sold']
vbs20.plot(kind='barh', x='Vendor Name',y='Total Bottles Sold',title='Top 20 Iowan Vendor Bottle Sales in 10 Millions (2012-2017)',figsize=[6,6])
#shows the top vendors by bottles
vs = df.groupby('vendor_name')[['sale_dollars']].sum()
vs = vs.sort_values('sale_dollars',ascending=False)
vs = vs.reset_index()
vs20 = vs.head(n=20)
vs20.columns = ['Vendor Name', 'Total Sales']
vs20.plot(kind='barh', x='Vendor Name',y='Total Sales',title='Top 20 Iowan Vendor Total Sales in $100 Millions (2012-2017)',figsize=[6,6])
totalSales = df.groupby(['item_description'])[['sale_dollars']].sum()
#total sales $$ by item
totalSales = totalSales.reset_index()
#reset the index
orderedSales = totalSales.sort_values('sale_dollars',ascending=False)
#order by sales
toptwentySales = orderedSales.head(n=20)
toptwentySales.columns = ['Product','Total Sales']
toptwentySales.plot(kind='barh', x='Product',y='Total Sales',title='Top 20 Product Total Sales in $10 Millions (2012-2017)',figsize=[6,6])
totalBottles = df.groupby(['item_description'])[['bottles_sold']].sum()
totalBottles = totalBottles.reset_index()
orderedBottles = totalBottles.sort_values('bottles_sold',ascending=False)
#orderbottles
totalBottles20 = orderedBottles.head(n=20)
totalBottles20.columns = ['Product', 'Total Bottles']
totalBottles20.plot(kind='barh', x='Product',y='Total Bottles',title='Top 20 Product Total Bottles Sold by Product (2012-2017)',figsize=[6,6])
bottlesByYear = df.groupby(['year'])[['bottles_sold']].sum()
bottlesByYear
bottlesByYear = bottlesByYear.reset_index()
bottlesByYear.plot.line(x='year',y='bottles_sold')
#why is there a drop off in sales?
stores2017 = df.groupby(['year'])[['store_number']].count()
#group by store number to see who has reported
stores2017
#significant drop off in 2017, possibly not all stores reporting
storeSales = df.groupby(['year','store_number'])[['sale_dollars']].sum()
storeSalesOrdered = storeSales.sort_values('sale_dollars',ascending=False)
#order by top stores with sales
storeSalesOrdered
#show sales by store number -No Show 2633 not reported in 2017, Show 4829 not reporting, and others
df.loc[df['store_number']==3385]
#search for store names - HyVee, Central City, Sams Club top sellers
storeSales2017 = df.groupby(['year','store_number'])[['sale_dollars']].sum()
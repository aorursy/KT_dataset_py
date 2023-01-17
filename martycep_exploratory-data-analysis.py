# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandasql import sqldf 

import plotly.express as px 

from plotly.subplots import make_subplots

from datetime import datetime 

import plotly.graph_objects as go



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# create dataframes 

list_orders = pd.read_csv("../input/ecommerce-data/List of Orders.csv", encoding='utf-8')

order_details = pd.read_csv("../input/ecommerce-data/Order Details.csv", encoding='utf-8')

sales_target = pd.read_csv("../input/ecommerce-data/Sales target.csv", encoding='utf-8')



#preprocessing 



#renamed columns for easier querying

list_orders.rename(columns={'Order ID':'ID', 'Order Date':'Date'}, inplace=True)

order_details.rename(columns={'Order ID':'ID', 'Sub-Category':'Sub_Category'}, inplace=True)

sales_target.rename(columns={'Month of Order Date':'MOD'}, inplace=True)



#drop missing values 

list_orders.dropna(inplace=True)

order_details.dropna(inplace=True)

sales_target.dropna(inplace=True)

#Seasonality of Purchases and Profit 



#Profit-Per-Item

order_details['PPI'] = order_details['Profit'] / order_details['Quantity']

orders_merged = sqldf("SELECT A.*,B.* FROM list_orders AS A LEFT JOIN order_details AS B on A.ID = B.ID")

orders_merged['Month'] = orders_merged['Date']



orders_merged['Date'] = pd.to_datetime(orders_merged.Date)
#Table 

orders_merged['Month'] = pd.to_datetime(orders_merged['Month']).dt.to_period('M').values





orders_merged['Month'] = orders_merged['Month'].astype(str)



pp_merged = sqldf("SELECT Month, SUM(Quantity) AS Items_Sold, SUM(Profit) AS Profit, SUM(PPI) AS PPI from orders_merged GROUP BY Month")

pp_table = go.Figure(data=[go.Table(header=dict(values=['Month', 'Items_Sold', 'Profit', 'PPI']),

                 cells=dict(values=[pp_merged['Month'], pp_merged['Items_Sold'],pp_merged['Profit'], pp_merged['PPI']]))

                     ])

pp_table.show()


#Line Charts 

Purchase_Dates= sqldf("SELECT Date, COUNT(ID) AS Purchases from orders_merged GROUP BY Date")

multi_purchases = sqldf("SELECT Date, Category, COUNT(ID) AS Purchases from orders_merged GROUP BY Date, Category")



Profit_Dates = sqldf("SELECT Date, SUM(Profit) AS Profits from orders_merged GROUP BY Date")

multi_profits = sqldf("SELECT Date, Category, SUM(Profit) AS Profits from orders_merged GROUP BY Date, Category")



Revenue_Dates = sqldf("SELECT Date, SUM(Amount) AS Revenue from orders_merged GROUP BY Date")

multi_revenues = sqldf("SELECT Date, Category, SUM(Amount) AS Revenue from orders_merged GROUP BY Date, Category")



#purchase

multi_prfig = px.line(multi_purchases, x="Date", y="Purchases", title='Purchases Over Time', color="Category")

multi_prfig.add_trace(go.Scatter(x=Purchase_Dates.Date, y=Purchase_Dates.Purchases,

                    mode='lines',

                    name='Total Purchases', line=dict(color='orange', width=1.5, dash='dash')))

multi_prfig.update_layout(autosize=False,width=1000, height=500)

multi_prfig.show()



#profit

multi_pfig = px.line(multi_profits, x="Date", y="Profits", title='Profits Over Time', color="Category")

multi_pfig.add_trace(go.Scatter(x=Profit_Dates.Date, y=Profit_Dates.Profits,

                    mode='lines',

                    name='Total Profit', line=dict(color='orange', width=1.5, dash='dash')))

multi_pfig.update_layout(autosize=False,width=1000, height=500)

multi_pfig.show()



#revenue 

multi_rfig = px.line(multi_revenues, x="Date", y="Revenue", title='Revenues Over Time', color="Category")

multi_rfig.add_trace(go.Scatter(x=Revenue_Dates.Date, y=Revenue_Dates.Revenue,

                    mode='lines',

                    name='Total Revenue', line=dict(color='orange', width=1.5, dash='dash')))

multi_rfig.update_layout(autosize=False,width=1000, height=500)

multi_rfig.show()
#Items sold in each category 

cat_sold = sqldf("SELECT Category, SUM(Quantity) AS Items_Sold from orders_merged GROUP BY Category")

bar = px.bar(cat_sold, x="Category", y="Items_Sold", title="Category Sales")

bar.show()



#Items sold in each subcategory

sub_sold = sqldf("SELECT Sub_Category, SUM(Quantity) AS Items_Sold from orders_merged GROUP BY Sub_Category")

barh = px.bar(sub_sold, x="Items_Sold", y="Sub_Category", orientation='h', color='Sub_Category', title="Subcategory Sales")

barh.show()



#Categories and PPI

def hist_dfs(a,b,c):

    hdf = sqldf("SELECT PPI from %s WHERE %s='%s'"%(a,b,c))

    return hdf



cat_ppi = sqldf("SELECT Category, AVG(PPI) AS AVG_PPI from orders_merged GROUP BY Category ORDER BY Avg_PPI DESC")



cat_0 = hist_dfs('order_details', 'Category','Clothing')

cat_1 = hist_dfs('order_details', 'Category','Furniture')

cat_2 = hist_dfs('order_details', 'Category','Electronics')



x0 = cat_0['PPI']

x1= cat_1['PPI']

x2= cat_2['PPI']



PPI_Hist = make_subplots(rows=3, cols=1, subplot_titles=("Clothing", "Furniture", "Electronics"), shared_xaxes=True)



Clothing = go.Histogram(x=x0, nbinsx=20)

Furniture = go.Histogram(x=x1, nbinsx =20)

Electronics = go.Histogram(x=x2, nbinsx=20)



PPI_Hist.append_trace(Clothing, 1, 1)

PPI_Hist.append_trace(Furniture, 2, 1)

PPI_Hist.append_trace(Electronics, 3, 1)



PPI_Hist.update_layout(title_text="PPI Spread Across Categories")



PPI_Hist.show()



PPI_Bar = px.bar(cat_ppi, x="Category", y="AVG_PPI", color="Category", title="Category Average PPI")

PPI_Bar.show()



clothing_sub = sqldf("SELECT Profit, Sub_Category, Category, PPI from orders_merged WHERE CATEGORY ='Clothing'")

electronics_sub = sqldf("SELECT Profit, Sub_Category, Category, PPI from orders_merged WHERE CATEGORY ='Electronics'")

furniture_sub = sqldf("SELECT Profit, Sub_Category, Category, PPI from orders_merged WHERE CATEGORY ='Furniture'")



# for the number of subcategories create a histogram 

clothing_sh = px.histogram(clothing_sub, x='PPI', color='Sub_Category', facet_row='Sub_Category', facet_col='Category')

clothing_sh.show()



electronics_sh = px.histogram(electronics_sub, x='PPI', color='Sub_Category', facet_row='Sub_Category', facet_col='Category')

electronics_sh.show()



furniture_sh = px.histogram(furniture_sub, x='PPI', color='Sub_Category', facet_row='Sub_Category', facet_col='Category')

furniture_sh.show()



#Subcategories and PPI

sub_ppi = sqldf("SELECT Sub_Category, AVG(PPI) AS AVG_PPI from orders_merged GROUP BY Sub_Category ORDER BY Avg_PPI DESC")

sub_names = order_details.Sub_Category.unique()

    

SPPI_Bar = px.bar(sub_ppi, x="Sub_Category", y="AVG_PPI", color="Sub_Category", title="Subcategory Average PPI")

SPPI_Bar.show()


#Categories and Profit 

cat_profit = sqldf("SELECT Category, SUM(Profit) AS Sum_Profit from orders_merged GROUP BY Category ORDER BY SUM_Profit DESC")

cat_pie = px.pie(cat_profit, values="Sum_Profit", names="Category", title="Category Profit Percentages")

cat_pie.show()

cat_bar = px.bar(cat_profit, x="Category", y="Sum_Profit", title="Category Profits")

cat_bar.show()





#Subcategories and Profit 

sub_profit = sqldf("SELECT Sub_Category, SUM(Profit) AS Sum_Profit from orders_merged GROUP BY Sub_Category ORDER BY SUM_Profit DESC")

sub_pie = px.pie(sub_profit, values="Sum_Profit", names="Sub_Category", title="Subcategory Profit Percentages")

sub_pie.show()

sub_bar = px.bar(sub_profit, x="Sub_Category", y="Sum_Profit", title="Category Profits", color="Sub_Category")



sub_bar.show()

#Location Variables

#Profits 

state_profit = sqldf('SELECT State, SUM(Profit) AS Total_Profit from orders_merged GROUP BY State ORDER BY Total_Profit DESC')

state_pbar = px.bar(state_profit[:5], x="State", y="Total_Profit", title="Best 5 State Profits", color="State", color_discrete_sequence=["#27ae60"])

state_pbar.show()



wstate_pbar = px.bar(state_profit.tail(5), x="State", y="Total_Profit", title="Worst 5 State Profits", color="State", color_discrete_sequence=["#c0392b"])

wstate_pbar.show()



city_profit = sqldf('SELECT City, SUM(Profit) AS Total_Profit from orders_merged GROUP BY City ORDER BY Total_Profit DESC')

city_pbar = px.bar(city_profit[:5], x="City", y="Total_Profit", title="Best 5 City Profits", color="City", color_discrete_sequence=["#27ae60"])

city_pbar.show()



wcity_pbar = px.bar(city_profit.tail(5), x="City", y="Total_Profit", title="Worst 5 City Profits", color="City", color_discrete_sequence=["#c0392b"])

wcity_pbar.show()



#Orders

state_orders = sqldf("SELECT State, SUM(Quantity) AS Items_Sold from orders_merged GROUP BY State ORDER BY Items_Sold DESC")

state_obar = px.bar(state_orders[:5], x="State", y="Items_Sold", title="5 Most In-Demand States", color="State", color_discrete_sequence=["#27ae60"])

state_obar.show()



wstate_obar = px.bar(state_orders.tail(5), x="State", y="Items_Sold", title="5 Least In-Demand States", color="State", color_discrete_sequence=["#c0392b"])

wstate_obar.show()





city_orders = sqldf("SELECT City, SUM(Quantity) AS Items_Sold from orders_merged GROUP BY City ORDER BY Items_Sold DESC")

city_obar = px.bar(city_orders[:5], x="City", y="Items_Sold", title="5 Most In-Demand Cities", color="City", color_discrete_sequence=["#27ae60"])

city_obar.show()



wcity_obar = px.bar(city_orders.tail(5), x="City", y="Items_Sold", title="5 Least In-Demand Cities", color="City", color_discrete_sequence=["#c0392b"])

wcity_obar.show()
sales_target['MOD'] = pd.to_datetime(sales_target['MOD'], format='%b-%y')

sales_target['MOD'] = pd.to_datetime(sales_target['MOD']).dt.to_period('M').values

sales_target['MOD'] = sales_target['MOD'].astype(str)



new_join = sqldf("SELECT A.*,B.* FROM list_orders AS A LEFT JOIN order_details AS B on A.ID = B.ID")

new_join['Month'] = pd.to_datetime(orders_merged['Date']).dt.to_period('M').values

new_join['Month'] = new_join['Month'].astype(str)



actual_sales = sqldf("SELECT Month, Category, SUM(Amount) AS Sales_Amount from new_join GROUP BY Month, Category")



sales_group = sqldf("SELECT * from sales_target GROUP BY MOD, Category")



sales = sqldf("SELECT A.*,B.Target FROM actual_sales AS A JOIN sales_group AS B on A.Month = B.MOD AND A.Category = B.Category")

sales['Difference'] = sales.Sales_Amount - sales.Target



quota_met = sqldf("SELECT * from sales WHERE Difference >= 0")

total_actual= sqldf("SELECT Month, SUM(Amount) AS Sales_Amount from new_join GROUP BY Month")

total_difference = sqldf("SELECT Month, SUM(Difference) AS Difference from sales GROUP BY Month")

sales_table = go.Figure(data=[go.Table(header=dict(values=['Month', 'Category', 'Sales_Amount', 'Target','Difference']),

                 cells=dict(values=[sales['Month'], sales['Category'],sales['Sales_Amount'], sales['Target'], sales['Difference'] ]))

                     ])

sales_table.show()



multi_sfig = px.line(sales, x="Month", y="Sales_Amount", title='Sales Over Time', color="Category")

multi_sfig.add_trace(go.Scatter(x=total_actual.Month, y=total_actual.Sales_Amount,

                    mode='lines',

                    name='Total Sales', line=dict(color='orange', width=1.5, dash='dash')))

multi_sfig.update_layout(autosize=False,width=1000, height=500)

multi_sfig.show()



multi_dfig = px.line(sales, x="Month", y="Difference", title='Difference Between Sales And Target', color="Category")

multi_dfig.add_trace(go.Scatter(x=total_difference.Month, y=total_difference.Difference,

                    mode='lines',

                    name='Total Difference', line=dict(color='orange', width=1.5, dash='dash')))



multi_dfig.update_layout(autosize=False,width=1000, height=500)

multi_dfig.show()
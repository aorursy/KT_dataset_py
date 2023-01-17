import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(os.listdir("../input"))
data_path_training = '../input/train.csv'
data_path_testing = '../input/test.csv'

data_to_train = pd.read_csv(data_path_training)
print(data_to_train.head())
df_train = pd.DataFrame(data_to_train)
df_train.describe()
df_train_table_store_item_date = pd.pivot_table(df_train,index=['store'],columns=['item','date'],values='sales')
df_train_table_store_item_date.head(10)
df_train_table_store_date_item = pd.pivot_table(df_train,index=['store'],columns=['date','item'],values='sales')
df_train_table_store_date_item.head()
plt.figure(figsize=(15,5)) #(width,height)

plt.subplot(1,3,1) #(row,col,index_in_this_matrix)
store_sales = pd.pivot_table(df_train,index=['store'],values='sales',aggfunc=[np.max])
plt.grid()
#plt.grid(color='r', linestyle='-', linewidth=1)
plt.title("Max Sales per store")
plt.plot(store_sales, 'ro')

plt.subplot(1,3,2) #(row,col,index_in_this_matrix)
store_sales = pd.pivot_table(df_train,index=['store'],values='sales',aggfunc=[np.mean])
plt.grid()
plt.title("Average Sales per store")
plt.plot(store_sales, 'gs')

plt.subplot(1,3,3) #(row,col,index_in_this_matrix)
plt.grid()
plt.title("Sale hits per store")
plt.scatter(df_train.store,df_train.sales)
bins = [10, 50, 100, 150, 200, 250]
plt.grid()
plt.hist(df_train.sales, bins, normed=1, histtype='bar', rwidth=0.8)
pivot_table_store_sales = pd.pivot_table(df_train,index=["store"],values=["sales"],aggfunc=[np.sum,np.mean])
print(pivot_table_store_sales)
print()
print(pivot_table_store_sales.max(axis=0))

# plt.grid()
# plt.plot(pivot_table_store_sales["sum"], 'ro')

pivot_table_store_sales["sum"].plot(kind="bar",legend="Sum", figsize=(20,5)).grid()
pivot_table_item_sales = pd.pivot_table(df_train,index=["item"],values=["sales"],aggfunc=[np.sum,np.mean])
print(pivot_table_item_sales.head(10))


# plt.grid()
# plt.plot(pivot_table_item_sales["sum"], 'bs')

pivot_table_item_sales["sum"].plot(kind="bar",legend="Sum", figsize=(30,10)).grid()
time_plot_store_performance = pd.pivot_table(df_train,index=['date'],columns=['store',],values='sales',aggfunc=np.sum)
time_plot_store_performance.head(15)
time_plot_items_performance = pd.pivot_table(df_train,index=['date'],columns=['item',],values='sales',aggfunc=np.sum)
time_plot_items_performance.head()
color_map = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
# for store in range(10):
#     performance_of_store = pd.pivot_table(df_train[df_train.store==(store+1)],index=['date'],columns=['store',],values='sales',aggfunc=np.sum)
#     plt.plot(performance_of_store, color=color_map[store])

plt.figure(figsize=(30,10)) #(width,height)
plt.title("Store'sale pattern")
# plt.grid()
for store in range(10):
    performance_of_store = pd.pivot_table(df_train[df_train.store==(store+1)],index=['date'],columns=['store',],values='sales',aggfunc=np.sum)
    plt.plot(performance_of_store, color=color_map[store],label="Store-"+str(store+1))
plt.legend(loc='upper left', shadow=True)
plt.figure(figsize=(30,10)) #(width,height)
plt.title("Item's sale pattern")
# plt.grid()
for item in range(10): #For now lets look at 10 items
    performance_of_item = pd.pivot_table(df_train[df_train.item==(item+1)],index=['date'],columns=['item',],values='sales',aggfunc=np.sum)
    plt.plot(performance_of_item, color=color_map[item%10],label="Item-"+str(item+1))
plt.legend(loc='upper left', shadow=True)
store_id = 5
sale_performance_of_store = pd.pivot_table(df_train[df_train.store==store_id],index=['date'],columns=['store',],values='sales',aggfunc=np.sum)
plt.title("Store-" + str(store_id) + "'s sale pattern")
plt.plot(sale_performance_of_store, 'bo',label="Store-"+str(store_id))
item_id = 2
sale_performance_of_item = pd.pivot_table(df_train[df_train.item==item_id],index=['date'],columns=['item',],values='sales',aggfunc=np.sum)
plt.title("Item-" + str(item_id) + "'s sale pattern")
plt.plot(sale_performance_of_item, 'go',label="Item-"+str(item_id))

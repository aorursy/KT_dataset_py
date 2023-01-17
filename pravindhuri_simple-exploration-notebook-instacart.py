import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

color = sns.color_palette()



%matplotlib inline
INPUT_FOLDER='/Users/pd186040/Documents/Kaggle/Instacart/'

print ('File Sizes:')

for f in os.listdir(INPUT_FOLDER):

    if 'zip' not in f:

       print (f.ljust(30) + str(round(os.path.getsize(INPUT_FOLDER +  f) / 1000, 2)) + ' KB')
order_products_train_df = pd.read_csv("/Users/pd186040/Documents/Kaggle/Instacart/order_products__train.csv")

order_products_prior_df = pd.read_csv("/Users/pd186040/Documents/Kaggle/Instacart/order_products__prior.csv")

orders_df = pd.read_csv("/Users/pd186040/Documents/Kaggle/Instacart/orders.csv")

products_df = pd.read_csv("/Users/pd186040/Documents/Kaggle/Instacart/products.csv")

aisles_df = pd.read_csv("/Users/pd186040/Documents/Kaggle/Instacart/aisles.csv")

departments_df = pd.read_csv("/Users/pd186040/Documents/Kaggle/Instacart/departments.csv")
print("The orders_df size is :", orders_df.shape)
orders_df.head(20)
print("The order_products_prior_df size is : ", order_products_prior_df.shape)
order_products_prior_df.head()
print("The order_products_train_df size is : ", order_products_train_df.shape)
order_products_train_df.head()
print("The products_df size is :", products_df.shape)
products_df.head()
print("The aisles_df size is :", aisles_df.shape)
aisles_df.head()
print("The departments_df size is :", departments_df.shape)
departments_df.head()
#checking for missing values

total=orders_df.isnull().sum()

total
#checking for the percentage

percentage=total/orders_df.isnull().count()

percentage
missing_value_table_orders = pd.concat([total,percentage],keys=['Total','Percentage'],axis=1)

missing_value_table_orders
orders_df_new=orders_df[orders_df['days_since_prior_order'].notnull()]

orders_df_new.head()
#aisles

total_a=aisles_df.isnull().count()

total_a
percentage_a=total_a/aisles_df.isnull().count()

percentage_a
missing_value_table_aisles = pd.concat([total_a, percentage_a],keys=['Total','Percentage'],axis=1)

missing_value_table_aisles
#departments

total_d=departments_df.isnull().count()

total_d
percentage_d=total_d/departments_df.isnull().count()

percentage_d
missing_value_table_departments = pd.concat([total_d,percentage_d],keys=['Total','Percentage'],axis=1)

missing_value_table_departments
#orders_prior

total_order_p_p=order_products_prior_df.isnull().sum()

total_order_p_p
percentage_order_p_p=total_order_p_p/order_products_prior_df.isnull().count()

percentage_order_p_p
missing_value_table_order_p_p = pd.concat([total_order_p_p,percentage_order_p_p],keys=['Total','Percentage'],axis=1)

missing_value_table_order_p_p
#order_train

total_order_train=order_products_train_df.isnull().sum()

total_order_train
percentage_order_train=total_order_train/order_products_train_df.isnull().count()

percentage_order_train
missing_value_table_order_train = pd.concat([total_order_train,percentage_order_train],keys=['Total','Percentage'],axis=1)

missing_value_table_order_train
#products

total_products=products_df.isnull().sum()

total_products
percentage_products=total_products/products_df.isnull().count()

percentage_products
missing_value_table_products = pd.concat([total_products,percentage_products],keys=['Total','Percentage'],axis=1)

missing_value_table_products
def get_unique_count(x):

    return len(np.unique(x))





cnt_eval = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)

cnt_eval
plt.figure(figsize=(12,8))

sns.barplot(cnt_eval.index, cnt_eval.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Eval set type', fontsize=12)

plt.title('Count of rows in each dataset', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
count=orders_df['eval_set'].value_counts()

count
plt.figure(figsize=(12,8))

sns.barplot(count.index, count.values)

plt.ylabel('Number of Occurrences in the dataset', fontsize=14)

plt.xlabel('Evaluation set type', fontsize=14)

plt.title('Eval_set breakdown in orders dataset', fontsize=16)
cnt_orders = orders_df.groupby("user_id")["order_number"].aggregate(np.max).reset_index()

cnt_orders = cnt_orders.order_number.value_counts()

plt.figure(figsize=(12,8))

sns.barplot(cnt_orders.index, cnt_orders.values, color=color[4])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Maximum order number', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="order_dow", data=orders_df, color=color[3])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Day of week', fontsize=12)

plt.title("Frequency of order by week day", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="order_hour_of_day", data=orders_df, color=color[5])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Hour of day', fontsize=12)

plt.title("Frequency of order by hour of day", fontsize=15)

plt.show()
grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()

grouped_df.head()
grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')

grouped_df
plt.figure(figsize=(12,8))

sns.heatmap(grouped_df)

plt.title("Frequency of Day of week Vs Hour of day")

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="days_since_prior_order", data=orders_df, color=color[2])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Days since prior order', fontsize=12)

plt.title("Frequency distribution by days since prior order", fontsize=15)

plt.show()
# percentage of re-orders in orders_products_prior

print("Percent of reorders in prior set:") 

print(order_products_prior_df.reordered.sum() / len(order_products_prior_df))
# percentage of re-orders in orders_products_train

print("Percent of reorders in train set:") 

print(order_products_train_df.reordered.sum() / len(order_products_train_df))
#merging order_products_prior and products

order_products_prior_df_merged = pd.merge(order_products_prior_df, products_df, on='product_id', how='left')



#merging op_merged with aisles

order_products_prior_df_merged = pd.merge(order_products_prior_df_merged, aisles_df, on='aisle_id', how='left')



#merging the new op_prior_merged with departments

order_products_prior_df_merged = pd.merge(order_products_prior_df_merged, departments_df, on='department_id', how='left')
order_products_prior_df_merged.head()
cnt_srs = order_products_prior_df_merged['product_name'].value_counts().reset_index().head(10)

cnt_srs.columns = ['product_name', 'frequency_count']

cnt_srs
cnt_srs = cnt_srs.groupby(['product_name']).sum()['frequency_count'].sort_values(ascending=False)

sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(12, 10))

sns.barplot(cnt_srs.index, cnt_srs.values)

plt.xticks(rotation='vertical')

plt.ylabel('Number of Reorders', fontsize=13)

plt.xlabel('Most ordered Products', fontsize=13)

plt.show()
cnt_aisle = order_products_prior_df_merged['aisle'].value_counts().head(20)

cnt_aisle
plt.figure(figsize=(12,8))

sns.barplot(cnt_aisle.index, cnt_aisle.values, alpha=0.8, color=color[5])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Aisle', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
cnt_aisle = order_products_prior_df_merged['department'].value_counts().head(20)

cnt_aisle
plt.figure(figsize=(12,8))

sns.barplot(cnt_aisle.index, cnt_aisle.values, alpha=0.8, color=color[2])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Departments', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
grouped =order_products_prior_df_merged.groupby(["department", "aisle"])["product_id"].aggregate({'Total_products': 'count'}).reset_index()

grouped.sort_values(by='Total_products', ascending=False, inplace=True)

grouped.head()
fig, axes = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=1.4))

for (aisle, group), ax in zip(grouped.groupby(["department"]), axes.flatten()):

    g = sns.barplot(group.aisle, group.Total_products , ax=ax)

    ax.set(xlabel = "Aisles", ylabel=" Number of products")

    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)

    ax.set_title(aisle, fontsize=15)
#merge order_product_prior with orders 

merged_reorders = pd.merge(order_products_prior_df, orders_df, on='order_id', how='left')

merged_reorders.head()
count_reordered = merged_reorders['reordered'].value_counts()

count_reordered
plt.figure(figsize=(6,12))

sns.barplot(count_reordered.index, count_reordered.values)

plt.ylabel('Frequencies', fontsize=14)

plt.xlabel('Reordered', fontsize=4)

plt.show()
#finding reorders against day of the week

grouped_reorders_dow = merged_reorders.groupby(["order_dow"])["reordered"].aggregate("count").reset_index()

grouped_reorders_dow
plt.figure(figsize=(6,12))

sns.barplot(grouped_reorders_dow.order_dow, grouped_reorders_dow.reordered)

plt.ylabel('Total number of reordered products', fontsize=14)

plt.xlabel('order_day_of_week', fontsize=14)

plt.show()
#finding reorders against hour of the day

grouped_reorders = merged_reorders.groupby(["order_hour_of_day"])["reordered"].aggregate("count").reset_index()

grouped_reorders
plt.figure(figsize=(12,12))

sns.barplot(grouped_reorders.order_hour_of_day, grouped_reorders.reordered)

plt.ylabel('Total number of reordered products', fontsize=14)

plt.xlabel('order_hour_of_day', fontsize=14)

plt.show()
merged1 = pd.merge(order_products_train_df, orders_df, on='order_id', how='left')

merged1.head()
df_merged1 = pd.merge(merged1, products_df, on='product_id', how='left')

df_merged1.head()
#merging all the datasets to get a final train dataset

df = pd.merge(df_merged1, departments_df, on='department_id', how='left')

df.head()
df_new = df.copy()

df_new.head()
del df['eval_set']
del df['add_to_cart_order']
df.head()


#Variable to be predicted

y=df['reordered']
del df['reordered']

del df['product_name']

del df['department']
df.head()


from sklearn.model_selection import train_test_split
Xtr, Xtest, ytr, ytest = train_test_split(df, y, test_size=0.30, random_state=5)

Xtr.shape

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import accuracy_score


#Logistic Regression model

clf=(LogisticRegression(C=0.02))
#fitting the model

clf.fit(Xtr, ytr)
#predictions

pred=clf.predict(Xtest)
pred
#accuracy score of Logistic Regression Model

print(accuracy_score(clf.predict(Xtest), ytest))
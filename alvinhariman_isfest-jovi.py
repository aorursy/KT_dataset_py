import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statistics
u = pd.read_csv("../input/users/users.csv", sep=";")

p = pd.read_csv("../input/products/products.csv", sep=";")

o = pd.read_csv("../input/orders/orders.csv", sep=";")

od = pd.read_csv("../input/orders-details/order_details.csv", sep=";")
display(u.head())

display(p.head())

display(o.head())

display(od.head())
print(u.shape)

print(p.shape)

print(o.shape)

print(od.shape)
print(u.drop_duplicates().shape)

print(p.drop_duplicates().shape)

print(o.drop_duplicates().shape)

print(od.drop_duplicates().shape)
p.category.unique()
p[p["desc_product"].duplicated(keep=False)].sort_values(by=["desc_product", "base_price"])
print(od[od["product_id"] == 28]["quantity"].sum())

print(od[od["product_id"] == 24]["quantity"].sum())

# print(od[od["product_id"] == 73]["quantity"].sum())
def drop_day(x):

    y = str(x)[0:7]

    return y
o["ym"] = o["created_at"].apply(drop_day)
ordering = list(o.ym.sort_values().unique())

time_label = ["Jan-19", "Feb-19", "Mar-19", "Apr-19", "Mei-19", "Jun-19", "Jul-19", 

              "Agu-19", "Sep-19", "Okt-19", "Nov-19", "Des-19", "Jan-20",

              "Feb-20", "Mar-20", "Apr-20", "Mei-20"]
plt.figure(figsize=(16,8))

ax = sns.countplot(x=o["ym"], order=ordering)

ax.set_xticklabels(time_label)

ax.set(xlabel = 'Waktu', ylabel = 'Jumlah Penjualan')

plt.title("Grafik Jumlah Penjualan terhadap Waktu")

plt.savefig("jumlah_penjualan.png")
p.groupby("category")["base_price"].describe().round(0).astype(int)
plt.figure(figsize=(12,8))

p.groupby("category")["base_price"].mean().plot.bar()
def get_domain(x):

    return str(x).split("@")[1]
u.email.apply(get_domain).unique()
len(list(u.email.apply(get_domain).unique()))
o.info()
o.head()
grouped = o.groupby("seller_id")["discount"].sum().reset_index()

grouped.sort_values("discount", ascending=False)
p.head()
def get_brand(x):

    return str(x).split(" ")[0]
p.desc_product.apply(get_brand).unique()
o
o[o.seller_id == o.buyer_id]
df1 = o.dropna(subset=['paid_at'])
paid_range = []

for i in range(df1.shape[0]):

    paid_range.append((pd.to_datetime(df1.iloc[i]['paid_at']) - pd.to_datetime(df1.iloc[i]['created_at'])).days)
print(statistics.mean(paid_range))

print(statistics.mode(paid_range))

print(statistics.median(paid_range))
df1 = o.dropna(subset=['delivery_at'])
delivery_range = []

for i in range(df1.shape[0]):

    delivery_range.append((pd.to_datetime(df1.iloc[i]['delivery_at']) - pd.to_datetime(df1.iloc[i]['paid_at'])).days)
print(statistics.mean(delivery_range))

print(statistics.mode(delivery_range))

print(statistics.median(delivery_range))
grouped = o.groupby("seller_id")["discount"].sum().reset_index()

grouped.sort_values("discount", ascending=False).head(10)
u
grouped = o.groupby("buyer_id")["total"].sum().reset_index()

grouped = grouped.sort_values("total", ascending=False).reset_index().head(10)

nama = []

total = []

for i in range(grouped.shape[0]):

    nama.append(u[u.user_id == grouped.iloc[i]['buyer_id']]['nama_user'].values[0])

    total.append(grouped.iloc[i]['total'])
df = pd.DataFrame({'nama':nama, 'total':total})

df
grouped = od.groupby("order_id")["quantity"].sum().reset_index()

grouped = grouped.sort_values("quantity", ascending=False).reset_index().head(10)

nama = []

quantity = []

for i in range(grouped.shape[0]):

    user_id = o[o.order_id == grouped.iloc[i]['order_id']]['buyer_id'].values[0]

    nama.append(u[u.user_id == user_id]['nama_user'].values[0])

    quantity.append(grouped.iloc[i]['quantity'])
df = pd.DataFrame({'nama':nama, 'quantity':quantity})

df
grouped = od.groupby("order_id")["quantity"].count().reset_index()

grouped = grouped.sort_values("quantity", ascending=False).reset_index().head(10)

nama = []

quantity = []

for i in range(grouped.shape[0]):

    user_id = o[o.order_id == grouped.iloc[i]['order_id']]['buyer_id'].values[0]

    nama.append(u[u.user_id == user_id]['nama_user'].values[0])

    quantity.append(grouped.iloc[i]['quantity'])
df = pd.DataFrame({'nama':nama, 'jumlah transaksi':quantity})

df
def get_discount(x):

    if int(x) == 0:

        return False

    else:

        return True
df1 = o.dropna(subset=['delivery_at'])
df1 = df1[df1["discount"].apply(get_discount)]

df1
paid_range = []

discount = []

for i in range(df1.shape[0]):

    discount.append(df1.iloc[i]['discount'])

    paid_range.append((pd.to_datetime(df1.iloc[i]['paid_at']) - pd.to_datetime(df1.iloc[i]['created_at'])).days)
statistics.mean(paid_range)
np.corrcoef(discount, paid_range)[0, 1]
discount = o[o["discount"].apply(get_discount)]['order_id'].count()

no_discount = o['order_id'].count() - discount
plt.bar(np.arange(2), [discount, no_discount], align='center', alpha=0.5)

plt.xticks(np.arange(2), ('discount', 'no_discount'))

plt.ylabel('jumlah transaksi')

plt.title('Pengaruh diskon dengan jumlah transaksi')



plt.show()
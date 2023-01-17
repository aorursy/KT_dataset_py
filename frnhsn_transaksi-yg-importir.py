# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import datetime

from pandas.tseries.offsets import DateOffset

import matplotlib.pyplot as plt

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
# Memuat dataset

order_detail = pd.read_json("/kaggle/input/transaksi-yg/yg_order_detail.json")

order = pd.read_json("/kaggle/input/transaksi-yg/yg_order.json")

status = pd.read_csv("/kaggle/input/transaksi-yg/yg_status.csv")

invoice = pd.read_json("/kaggle/input/transaksi-yg/com_invoice.json")

invoice_detail = pd.read_json("/kaggle/input/transaksi-yg/com_invoice_detail.json")

invoice_list = pd.read_csv("/kaggle/input/transaksi-yg/com_invoice_list.csv")

refund = pd.read_json("/kaggle/input/transaksi-yg/account_trx.json")

product = pd.read_json("/kaggle/input/transaksi-yg/yg_product_detail.json")

category = pd.read_json("/kaggle/input/transaksi-yg/category.json")





subs = pd.read_csv('/kaggle/input/importir/subs.csv')

user = pd.read_csv('/kaggle/input/importir/user.csv')

all_users = pd.read_json('/kaggle/input/importir/org_users.json')



all_users.rename(columns={'id':'user_id'},inplace=True)

subs.rename(columns={'id':'subs_id'},inplace=True)

order.rename(columns={'id':'order_id'},inplace=True)



# Penentuan tipe data

order.flag = order.flag.astype('category')

order.currency = order.currency.astype('category')

order.logistic = order.logistic.astype('category')

order.warehouse_delivery_currency = order.warehouse_delivery_currency.astype('category')

status['created_at'] = pd.to_datetime(status['created_at'], errors='coerce')

order = order.set_index('order_id').join(order_detail.groupby('yg_order_id').delivery_type.min()).reset_index()

order['mobile'] = order.token != ''



order.loc[:,'created_at'] = pd.to_datetime(order['created_at'], errors='coerce')

order_detail.loc[:,'created_at'] = pd.to_datetime(order_detail['created_at'], errors='coerce')

status.loc[:,'created_at'] = pd.to_datetime(status['created_at'], errors='coerce')

invoice_detail.loc[:,'created_at'] = pd.to_datetime(invoice_detail['created_at'], errors='coerce')



# Data slicing

order = order[order.created_at < np.datetime64('2020-04-06')]

order_detail = order_detail[order_detail.created_at < np.datetime64('2020-04-06')]

status = status[status.created_at < np.datetime64('2020-04-06')]

invoice_detail = invoice_detail[invoice_detail.created_at < np.datetime64('2020-04-06')]



subs['email'] = subs.email.str.lower()

all_users['email'] = all_users.email.str.lower()



# Penyeragaman data

status['title'] = status.title.str.lower()

order.flag[order.flag == 'other'] = 'others'



# Membersihkan data status

status = status.replace({'title': {'partial in shipping': 'partial shipping'}})

status = status.groupby(['yg_order_id','title'])['created_at'].min()



# Menyingkirkan kolom yang tidak perlu

status = status.drop(columns=['message','updated_at','deleted_at'])

order = order.drop(columns=['delivery_fee','logistic_message','payment_method','is_official_1688'])



order = order.set_index('order_id').join(status.unstack()['customer paid'].rename('order_paid_at')).reset_index()

invoice_detail['amount'] = invoice_detail.type.map({'+': 1, '-':-1}) * invoice_detail.amount



# Tagihan pertama dibayar

invoice['bills_number'] = (invoice.order_id == 0).map({True: 1, False: 2})

paid_id = invoice_list[invoice_list.com_invoice_id.isin(

    invoice[(invoice['paid_at'].notnull()) & (invoice['order_id'] == 0)]['id'].to_list())]['yg_order_id'].to_list()

invoice_first_bill_paid = invoice[(invoice['paid_at'].notnull()) & (invoice['order_id'] == 0)]

invoice_first_bill_paid = invoice_first_bill_paid.append(invoice[invoice.order_id.isin(paid_id)])



invoice_paid_id = invoice[invoice.paid_at.notnull()]['id'].to_list()

invoice_detail_paid = invoice_detail[invoice_detail['com_invoice_id'].isin(invoice_paid_id)]

refund['yg_order_id'] = refund.yg_order_id.apply(lambda x: int(x) if np.isfinite(x) else x)



invoice['tagihan_produk'] = invoice.order_id == 0



# Refund

refund = refund[refund['transferred_at'].notnull()]

refund = refund[refund.order_type == 'yg']

refund = refund[['account_user_id','created_by', 'yg_order_id', 'amount', 

                 'admin_note', 'transferred_by', 'created_at']]
# Megubah total_price Free-Member jadi 0 rupiah

subs.loc[subs.package_name == 'Free-Member', 'total_price'] = 0



# Mengubah tipe data kolom paid_at, created_at pada dataset subs menjadi datetime

subs['paid_at'] = pd.to_datetime(subs['paid_at'], errors='coerce')

subs['created_at'] = pd.to_datetime(subs['created_at'], errors='coerce')



# Mengubah gold-3-tahun jadi Gold-3-tahun

subs.loc[subs.package_name == 'gold-3-tahun', 'package_name'] = "Gold-3-tahun"



# Menentukan waktu expired dari masing masing paket

subs.loc[subs.package_name == 'AnR-Basic-Plus', 'expired_at'] = subs['paid_at'] + DateOffset(months=6)

subs.loc[subs.package_name == 'AnR-Gold', 'expired_at'] = subs['paid_at'] + DateOffset(months=12)

subs.loc[subs.package_name == 'Basic-Harbolnas', 'expired_at'] = subs['paid_at'] + DateOffset(months=1)

subs.loc[subs.package_name == 'Basic-Plus', 'expired_at'] = subs['paid_at'] + DateOffset(months=1)

subs.loc[subs.package_name == 'Basic-Plus-24-month', 'expired_at'] = subs['paid_at'] + DateOffset(months=24)

subs.loc[subs.package_name == 'Goes to China', 'expired_at'] = subs['paid_at'] + DateOffset(months=36)

subs.loc[subs.package_name == 'Gold', 'expired_at'] = subs['paid_at'] + DateOffset(months=12)

subs.loc[subs.package_name == 'Membership 3 Tahun', 'expired_at'] = subs['paid_at'] + DateOffset(months=36)

subs.loc[subs.package_name == 'Silver', 'expired_at'] = np.datetime64('NaT')

subs.loc[subs.package_name == 'Silver', 'expired_at'] = np.datetime64('NaT')

subs['expired_at'] = pd.to_datetime(subs['expired_at'], format='%d/%m/%Y', errors='coerce')
order = order.set_index('order_id').join(order_detail.groupby(

    'yg_order_id')[['ship_from','ship_to']].min(), how='left').reset_index()

order = order.set_index('user_id').join(

    all_users[['user_id','email']].set_index('user_id'),how='left').reset_index()



cross_prod = order[['order_id','email','order_paid_at']].set_index('email').join(

    subs[['subs_id','email','package_name','paid_at','expired_at']].set_index('email'))



def find_package(df):

    result = df[(df.order_paid_at > df.paid_at) & \

                (df.order_paid_at <= df.expired_at)]['subs_id'].values

    if len(result) > 0:

        return int(result[0])

    else:

        None



result = cross_prod.groupby('order_id').apply(find_package)



order = order.set_index('order_id').join(result.rename('subs_id')).reset_index()

order = order.set_index('subs_id').join(subs.set_index('subs_id')['package_name']).reset_index()
order.set_index('order_id',inplace=True)



order.loc[order.package_name.isna(), 'package_name'] = order[

    order.package_name.isna()].reset_index().set_index('user_id').join(

    all_users.set_index('user_id')['membership_package'],

    how='left').reset_index().set_index('order_id')['membership_package'].replace('Gold','Gold-ORG')



order.reset_index(inplace=True)
mobile = pd.concat([order['mobile'].value_counts(),

           (order['mobile'].value_counts()/order['mobile'].count()*100).round(2)], axis = 1)

mobile.index = ['web','mobile']

mobile.columns = ['Jumlah order', 'Persen']

mobile
order['flag'].value_counts().to_frame().T
order['platform'] = order['mobile'].map({True: 'Mobile', False: 'Web'})

pd.crosstab(order['platform'],order['flag'])
order.product_price.describe().apply("{0:,.2f}".format)
fig, ax = plt.subplots(figsize=(12, 5))

g = sns.distplot(order.product_price, kde=False, norm_hist=False)

g.set_yticklabels(['{:,.0f}'.format(x) for x in g.get_yticks()])

g.set_xticklabels(['{:,.0f}'.format(x) + ' juta' for x in g.get_xticks()/1000000])

plt.show()
prod_price_cut = order

max_price = (round(prod_price_cut.product_price.max() / 10000000) + 1) * 10000000

interval_range = pd.interval_range(start=0, freq=10000000, end=max_price)

prod_price_cut['cut'] = pd.cut(prod_price_cut['product_price'], bins=interval_range)

qct = pd.DataFrame(prod_price_cut.groupby('cut').product_price.sum()).reset_index()

fig, ax = plt.subplots(figsize=(12, 5))

ax = sns.barplot(x='cut', y='product_price', data=qct, palette=sns.color_palette("muted", n_colors=1))

ax.set_yticklabels(['{:,.0f}'.format(x) + ' juta' for x in ax.get_yticks()/1000000])

plt.xticks(rotation=90)

plt.show()
order['paid_month_year'] = order.order_paid_at.dt.strftime('%Y-%m')

prod_price_group = order.groupby('paid_month_year')['product_price'].sum()

fig, ax = plt.subplots(figsize=(12, 5))

g = sns.lineplot(data=prod_price_group)

g.set_yticklabels(['{:,.0f}'.format(x) + ' juta' for x in g.get_yticks()/1000000])

plt.xticks(rotation=90)

plt.title = 'Pendapatan produk dari order'

plt.show()
inv = pd.DataFrame(invoice_detail_paid.groupby('tag')['amount'].sum())

inv['percent'] = (inv['amount'] / sum(inv['amount']) * 100).round(2)

inv.style.format("{:,.2f}")
tagihan_yang_tagihan_pertama_terbayar = invoice_detail[invoice_detail.com_invoice_id.isin(invoice_first_bill_paid.id)]

tagihan_yang_tagihan_pertama_terbayar.groupby(['order_id','tag']).amount.sum().unstack(1).count()
tagihan_inv = invoice_detail[invoice_detail.com_invoice_id.isin(

    invoice_first_bill_paid[invoice_first_bill_paid.paid_at.notnull()].id.to_list())].amount.sum()



print("Tagihan terbayar (tagihan pertama dan kedua): ", invoice_detail_paid.amount.sum())

print("Tagihan terbayar jika tagihan pertama terbayar: ", tagihan_inv)
invoice[invoice.order_id != 0].title.value_counts().head(20)
n_order_1 = order.order_paid_at.notnull().sum()

n_order_2 = len(invoice_list[invoice_list.com_invoice_id.isin(

    invoice[(invoice.paid_at.notnull()) & (invoice.order_id == 0)]['id'])

                ].yg_order_id.unique())



print("Jumlah order terbayar dari tabel order:", n_order_1)

print("Jumlah order terbayar dari tagihan pertama terbayar:", n_order_2)
invoice[invoice.paid_at.notnull()].tagihan_produk.value_counts()
pd.crosstab(invoice_first_bill_paid['bills_number'],invoice_first_bill_paid['paid_at'].notnull().rename('is_paid'))
pd.crosstab(invoice_first_bill_paid.bills_number,

            invoice_first_bill_paid.paid_at.notnull().rename('is_paid'),

            values=invoice_first_bill_paid.amount,

            aggfunc=np.sum).style.format("{:,.2f}")
# Perbandingan transaksi produk dari invoice dibanding order

print('Nilai transaksi produk dari invoice:', invoice_detail_paid[invoice_detail_paid.tag == 'product']['amount'].sum())

print('Nilai transaksi produk dari order:', order.product_price.sum())

print('Selisih:', order.product_price.sum() - invoice_detail_paid[invoice_detail_paid.tag == 'product']['amount'].sum())
refund.head()
print("Jumlah order refund:",len(refund.yg_order_id.unique()))

print("Nilai refund:",refund.amount.sum())
jenis_pengiriman = order.reset_index().groupby('delivery_type')[['order_id','product_price']].agg({

    'order_id': 'count',

    'product_price': 'sum',

})

jenis_pengiriman.rename(columns={

    'order_id': 'jumlah_order',

    'product_price': 'nilai_produk',

}).style.format("{:,.2f}")
pd.crosstab(order['ship_from'],order['ship_to'])
pd.crosstab(order['ship_from'],

            order['ship_to'],

            values=order['product_price'],

            aggfunc=np.sum).style.format("{:,.2f}")
tujan_pengiriman = order_detail[['address', 'province', 'city', 'district', 'post_code']]

tujan_pengiriman['count_'] = 1

group_pengiriman = tujan_pengiriman.groupby(['address', 'province', 'city', 'district', 'post_code'],

                         as_index=False).sum().sort_values(by='count_',ascending=False)

group_pengiriman.head(10)
group_provinsi = group_pengiriman.groupby('province').count_.sum()



prov = group_provinsi.nlargest(8)

prov = prov.append(pd.Series({'LAINNYA': sum(

    group_provinsi[group_provinsi.index.isin(group_provinsi.nlargest(8).index) == False])}))



fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.pie(prov.values, labels=prov.index, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  

plt.tight_layout()

plt.show()
status_ct = status.unstack().drop(columns=['changed to yg478','diganti yg826','link expired', 

                        'need confirmation','partial shipping','re-order','unpaid', 

                        'waitbuyerpay','waitbuyerreceive', 'waitsellersend'])



status_ct.index.name = 'order_id'



status_ct = pd.to_datetime(status_ct.stack(), errors='coerce').unstack()

status_td = pd.DataFrame(columns=[

    'supplier paid - customer paid',

    'customer paid - in warehouse china',

    'in warehouse china - in shipping',

    'in shipping - in warehouse indo',

    'in warehouse indo - done',    

])

status_td['supplier paid - customer paid'] = status_ct['supplier paid'] - status_ct['customer paid']

status_td['customer paid - in warehouse china'] = status_ct['in warehouse'] - status_ct['customer paid']

status_td['in warehouse china - in shipping'] = status_ct['in shipping'] - status_ct['in warehouse']

status_td['in shipping - in warehouse indo'] = status_ct['in warehouse indo'] - status_ct['in shipping']

status_td['in warehouse indo - done'] = status_ct['done'] - status_ct['in warehouse indo']

status_td.index = status_ct.index

status_td['delivery_type'] = status_td.join(order.set_index('order_id').delivery_type).delivery_type

status_td.loc[status_td.delivery_type == '', 'delivery_type'] = np.nan
status.reset_index(1).title.value_counts()
status_td.delivery_type.value_counts(dropna=False)
data = status_td.drop(columns=['delivery_type']).apply(lambda x: x.dt.days)

data = pd.DataFrame(data.stack()).reset_index(1)

data.columns = ['status', 'period']

data = data.join(status_td['delivery_type'])



fig, ax = plt.subplots(figsize=(12, 20))

g = sns.boxplot(x="status", y="period", hue='delivery_type', data=data.dropna(subset=['delivery_type']))

plt.xticks(rotation=90)

yrange = np.arange(round(min(data.period.dropna())/20-1)*20, round(max(data.period.dropna())/20+1)*20, 10.0)

plt.yticks(yrange)

plt.show()
status_td.describe()

pd.DataFrame(status_td.groupby('delivery_type').describe().stack())
invoice_detail_paid.groupby(['order_id','tag']).amount.sum().unstack(1).count()
dimensi = order[['cbm_total','weight','carton_total','dimension_height',

                 'dimension_width','dimension_length', 'delivery_type',

                 'order_paid_at']]



# dimensi.loc[:,'cbm_total'] = dimensi.dimension_height * dimensi.dimension_width * dimensi.dimension_length / 10 ** 6

mask = dimensi.select_dtypes(include=[np.number]).columns

dimensi.describe()
dimensi = dimensi[(dimensi[mask] > 0).all(axis=1)]

dimensi.loc[dimensi['delivery_type'] == '', 'delivery_type'] = np.nan

dimensi[mask].astype(bool).sum(axis=0)
dimensi['delivery_type'].value_counts(dropna=False)
fig = dimensi.hist(bins=20, figsize=(12, 10))
fig = plt.figure(figsize=(12, 5))

pplot = sns.pairplot(dimensi.drop(columns=['order_paid_at']), hue='delivery_type', diag_kind='hist')

plt.show()
fig, ax = plt.subplots(figsize=(12, 5))

bool_filter = [abs(x) < 1 for x in stats.zscore(dimensi['weight'])]

dimensi[bool_filter].dropna()

g = sns.boxplot(x='weight', y='delivery_type',data=dimensi)

plt.show()
fig, ax = plt.subplots(figsize=(12, 5))

g = sns.boxplot(x='dimension_width',y='delivery_type',data=dimensi)

plt.show()
fig, ax = plt.subplots(figsize=(12, 5))

g = sns.boxplot(x='dimension_length',y='delivery_type',data=dimensi)

plt.show()
fig, ax = plt.subplots(figsize=(12, 5))

g = sns.boxplot(x='dimension_height',y='delivery_type',data=dimensi)

plt.show()
fig, ax = plt.subplots(figsize=(12, 5))

g = sns.boxplot(x='cbm_total',y='delivery_type',data=dimensi)

plt.show()
fig, ax = plt.subplots(figsize=(12, 5))

g = sns.boxplot(x='carton_total',y='delivery_type',data=dimensi)

plt.show()
dimensi.groupby(

    ['delivery_type',

     pd.Grouper(key='order_paid_at', freq='D')]).sum().reset_index().groupby(

    'delivery_type').mean()
dimensi.groupby(

    ['delivery_type',

     pd.Grouper(key='order_paid_at', freq='M')]).sum().reset_index().groupby(

    'delivery_type').mean()
satu_bulan_lalu = np.datetime64('today') - np.timedelta64(30, 'D')



status_barang_sisa = status.unstack()

status_barang_sisa = status_barang_sisa[(status_barang_sisa['in warehouse indo'] < satu_bulan_lalu) &\

                   (status_barang_sisa['in warehouse indo'].notnull()) &\

                   (status_barang_sisa['done'].isnull()) &\

                   (status_barang_sisa['cancel'].isnull()) &\

                   (status_barang_sisa['cancel and refund'].isnull())

                  ]
len(status_barang_sisa['in warehouse indo'])
fig, ax = plt.subplots(figsize=(12, 5))

lama_barang_sisa = np.datetime64('today') - status_barang_sisa['in warehouse indo']

g = sns.distplot(lama_barang_sisa.dt.days, kde=False, norm_hist=False)

plt.show()
tagihan_barang_sisa
tagihan_barang_sisa = invoice_detail[invoice_detail.order_id.isin(status_barang_sisa.index)].set_index('com_invoice_id').join(

    invoice.set_index('id').paid_at.notnull())

tagihan_barang_sisa.rename(columns = {'paid_at':'is_paid'}, inplace = True)
tagihan_barang_sisa.groupby(['tag','is_paid']).amount.sum().unstack(0).style.format("{:,.2f}")
category.category_id = category.category_id.apply(lambda x: int(x) if len(x) > 0 else np.nan)
fig = product.hist(bins=20, figsize=(12, 10))
product.describe()
product.isna().sum().to_frame()
order_detail.rename(columns={'yg_product_id': 'product_id'},inplace=True)

order_detail = order_detail.set_index('product_id').join(product[['product_id', 'category_id', 'category_name_en', 'title_en', 'price_fix',

         'view', 'flag', 'is_flash_sale','weight_per_product']].set_index('product_id')).reset_index().rename(columns={'index':'product_id'})
order_detail.groupby('product_id')[

    ['id','quantity','price_total']].agg({

    'id':'count',

    'quantity':'sum',

    'price_total':'sum'}).rename(columns={

    'id':'order_count',

    'quantity':'total_quantity',

    'price_total':'total_product_price'}).sort_values(

    'total_quantity',ascending=False).join(product[

    ['product_id','title_en','category_name_en']].set_index(

    'product_id')).head(20).style.format({"total_product_price":"{:,.2f}"})
product
cat = category[['parent_id','category_id','name']].set_index('category_id')
# order_detail.groupby('category_id')[

#     ['id','quantity','price_total']].agg({

#     'id':'count',

#     'quantity':'sum',

#     'price_total':'sum'}).rename(columns={

#     'id':'order_count',

#     'quantity':'total_quantity',

#     'price_total':'total_product_price'}).sort_values(

#     'total_quantity',ascending=False).join(product[

#     ['product_id','title_en','category_name_en']].set_index(

#     'category_id')).head(20).style.format({"total_product_price":"{:,.2f}"})
# order_detail.groupby('category_id')[

#     ['id','quantity','price_total']].agg({

#     'id':'count',

#     'quantity':'sum',

#     'price_total':'sum'}).rename(columns={

#     'id':'order_count',

#     'quantity':'total_quantity',

#     'price_total':'total_product_price'}).sort_values(

#     'total_quantity',ascending=False).join(product[

#     ['product_id','category_name_en']].set_index(

#     'product_id'))
user_order_count = order.groupby('user_id')['product_price'].agg(

    ['sum','count']).rename(columns={'sum':'nilai_belanja_produk',

                                     'count':'jumlah_order'})
user_order_count.describe().style.format("{:,.2f}")
fig, ax = plt.subplots(figsize=(12, 5))

g = sns.distplot(user_order_count['nilai_belanja_produk'], kde=False, norm_hist=False)

g.set_xticklabels(['{:,.0f}'.format(x) + ' juta' for x in g.get_xticks()/1000000])

plt.show()
fig, ax = plt.subplots(figsize=(12, 5))

g = sns.distplot(user_order_count['jumlah_order'], kde=False, norm_hist=False)

plt.show()
grouped = order.groupby(['user_id','package_name'])['product_price'].agg(['sum','count']).reset_index(1)

pplot = sns.pairplot(grouped, hue='package_name', diag_kind = 'kde')

pplot.fig.set_size_inches(15, 10)
order.groupby('package_name')['product_price'].agg(['count','sum']).style.format("{:,.2f}")
df = order[['order_id','user_id','product_price','order_paid_at']]

df.rename(columns={'order_paid_at':'paid_date'}, inplace=True)

df.dropna(subset=['paid_date'], inplace=True)
df['order_period'] = df.paid_date.apply(lambda x: x.strftime('%Y-%m'))
df.set_index('user_id', inplace=True)

df['cohort_group'] = df.groupby(level=0)['paid_date'].min().apply(lambda x: x.strftime('%Y-%m'))

df.reset_index(inplace=True)



grouped = df.groupby(['cohort_group', 'order_period'])

cohorts = grouped.agg({'user_id': pd.Series.nunique,

                       'order_id': pd.Series.nunique,

                       'product_price': np.sum})



cohorts.rename(columns={'user_id': 'total_users',

                        'order_id': 'total_orders',

                        'product_price': 'product_charges'}, inplace=True)
def cohort_period(df):

    df['cohort_period'] = np.arange(len(df)) + 1

    return df



cohorts = cohorts.groupby(level=0).apply(cohort_period)
# reindex the DataFrame

cohorts.reset_index(inplace=True)

cohorts.set_index(['cohort_group', 'cohort_period'], inplace=True)



# create a Series holding the total size of each CohortGroup

cohort_group_size = cohorts['total_users'].groupby(level=0).first()
cohorts['total_users'].unstack(0).transpose()
user_retention = cohorts['total_users'].unstack(0).divide(cohort_group_size, axis=1)
user_retention.iloc[:,[0,1,2]].plot(figsize=(10,5))

# plt.title('Cohorts: User Retention')

plt.xticks(np.arange(1, 12.1, 1))

plt.xlim(1, 12)

plt.ylabel('% of Cohort Purchasing');
sns.set(style='white')



plt.figure(figsize=(12, 8))

# plt.title('Cohorts: User Retention')

sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%');
df = invoice_detail_paid.groupby(['order_id','tag'])['amount'].sum().unstack(1)

df['total_charges'] = df.sum(axis=1)

df = df[['delivery_fee', 'product','shipping_fee', 'warehouse_delivery_fee','total_charges']]

df.rename(columns={'product':'product_fee'},inplace=True)

df.dropna(inplace=True)

df = df[(df > 0).all(axis=1)]

df = df.join(order.set_index('order_id').delivery_type,how='left')

df = df[df.delivery_type == 'air']

df.drop(columns=['delivery_type'],inplace=True)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt  #for plotting purpose

from sklearn.linear_model import LinearRegression   #for implementing multiple linear regression

from sklearn.model_selection import train_test_split



# X = df[['delivery_fee', 'product_fee', 'shipping_fee', 'warehouse_delivery_fee']]

X = df[['product_fee', 'warehouse_delivery_fee']]

# X = df[['delivery_fee', 'product_fee','shipping_fee', 'warehouse_delivery_fee']]

y = df['total_charges']
df
sns.pairplot(df)
df.drop(columns='total_charges').corr()
# sns.scatterplot(x='product_fee',y='total_charges',data=df)

sns.regplot(x='product_fee',y='total_charges',data=df, order=1, ci=None)
sns.regplot(x='warehouse_delivery_fee',y='total_charges',data=df, order=1, ci=None)
# X = X['delivery_fee'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)



# X_train = X[-4000:].values 

# X_test = X[4000:].values 

# y_train = y[-4000:].values 

# y_test = y[4000:].values 



regressor = LinearRegression()

regressor.fit(X_train, y_train)



# regressor.coef_

coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])

coeff_df
y_pred = regressor.predict(X_test)

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1['diff'] = df1.Actual - df1.Predicted

df1.style.format("{:,.2f}")
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
abs(df1['diff']).describe().to_frame().style.format("{:,.2f}")
import matplotlib.ticker as ticker



fig, ax = plt.subplots(figsize=(20, 5))   

sns.scatterplot(x='product_fee',y='diff',data=df1.join(X_test.product_fee))

ax.xaxis.set_major_formatter(ticker.EngFormatter())

ax.yaxis.set_major_formatter(ticker.EngFormatter())
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import matplotlib.dates as mdates

import matplotlib.ticker as mticker

import seaborn as sns

sns.set()



%matplotlib inline
transaction_df = pd.read_csv("/kaggle/input/quantium-data-analytics-virtual-experience-program/Transactions.csv")

behaviour_df = pd.read_csv("/kaggle/input/quantium-data-analytics-virtual-experience-program/PurchaseBehaviour.csv")
transaction_df.head()
behaviour_df.head()
transaction_df.isna().sum()
behaviour_df.isna().sum()
def convert_to_datetime(num):

    dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + num - 2)

    return dt
%%time

transaction_df['DATE'] = transaction_df['DATE'].apply(convert_to_datetime)

transaction_df.head()
transaction_df["PROD_NAME"].unique()[:20]
def remove_prod(df):

    unwanted = "salsa"

    if unwanted in df["PROD_NAME"].lower().split():

        return df.name
%%time

drop_index = list(transaction_df.apply(remove_prod,axis=1))
%%time

#Making a copy of all transaction and updating the new transactions without unwanted products

old_df = transaction_df.copy()

transaction_df = transaction_df.drop([i for i in drop_index if ~np.isnan(i)])
print("Total Transaction: "+str(len(old_df)))

print("Total Salsa Products: "+str(len(old_df)-len(transaction_df)))

print("Transactions without Salsa: "+str(len(transaction_df)))
transaction_df.describe()
transaction_df[transaction_df.PROD_QTY > 100]
behaviour_df[behaviour_df.LYLTY_CARD_NBR == 226000]
transaction_df = transaction_df[transaction_df['PROD_QTY'] < 200].reset_index(drop=True)

transaction_df.describe()
def packet_size(grp):

    string = grp["PROD_NAME"]

    num = []

    for i in string:

        if i.isdigit():

            num.append(i)

    number = "".join(num)

    return int(number)
%%time

transaction_df["PACKET_SIZE"] = transaction_df.apply(packet_size,axis=1)

transaction_df.head()
print("Largest Packet Size: "+str(max(transaction_df["PACKET_SIZE"]))+"g")

print("Smallest Packet Size: "+str(min(transaction_df["PACKET_SIZE"]))+"g")
def Product_Company(grp):

    return grp["PROD_NAME"].split()[0]
%%time

transaction_df["BRAND"] = transaction_df.apply(Product_Company,axis=1)

transaction_df.head()
d = {'red':'RRD','ww':'WOOLWORTHS','ncc':'NATURAL','snbts':'SUNBITES','infzns':'INFUZIONS','smith':'SMITHS','dorito':'DORITOS','grain':'GRNWVES'}

transaction_df['BRAND'] = transaction_df['BRAND'].str.lower().replace(d).str.upper()

transaction_df.head()
date_sales = pd.DataFrame(transaction_df.groupby("DATE").agg({'TOT_SALES':'sum'}))
plt.style.use('seaborn')

fig = date_sales.plot(figsize=(18,8))

plt.ylabel("Total Sales",{'fontsize':15})

ax = plt.gca()

ax.xaxis.set_major_locator(mdates.MonthLocator())

ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))

ax.xaxis.set_major_formatter(mticker.NullFormatter())

ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b-%y"))
december = pd.DataFrame({'Date' : pd.date_range(start='2018-12-01', end='2018-12-31'),'sales' : np.zeros((31))}).set_index('Date')

december.sales = date_sales.loc[[i for i in december.index if i in date_sales.index]]

december.fillna(0,inplace=True)
fig = plt.figure(figsize=(18,6))

plt.bar(december.index,december.sales)

ax = plt.gca()

formatter = mdates.DateFormatter("%b-%d")

ax.xaxis.set_major_formatter(formatter)

locator = mdates.DayLocator()

ax.xaxis.set_major_locator(locator)

plt.ylabel("Total Sales",{'fontsize':15})

plt.xticks(rotation='vertical')

plt.show()
plt.xlabel('Number of Transactions',{'fontsize':15})

plt.ylabel('Brands',{'fontsize':15})

transaction_df.BRAND.value_counts().sort_values().plot(kind='barh',figsize=(18,8))
plt.xlabel('Number of Packets',{'fontsize':15})

plt.ylabel('Packet Size',{'fontsize':15})

transaction_df.PACKET_SIZE.value_counts().plot(kind='bar',figsize=(18,8))

plt.xticks(rotation='horizontal')

plt.show()
behaviour_df.head()
behaviour_df.nunique()
plt.figure(figsize=(7,7))

plt.title('Customers Valuation Distribution',{'fontsize': 15})

plt.pie(behaviour_df.PREMIUM_CUSTOMER.value_counts(),labels=behaviour_df.PREMIUM_CUSTOMER.value_counts().index)

plt.show()
plt.figure(figsize=(21,4))



plt.subplot(131)

plt.title('Lifestage Distribution of Budget Customers',{'fontsize': 15})

plt.ylabel("Number of Transactions",{"fontsize":12})

behaviour_df.groupby("PREMIUM_CUSTOMER").LIFESTAGE.value_counts()['Budget'].plot(kind='bar',fontsize=12)



plt.subplot(132)

plt.ylabel("Number of Transactions",{"fontsize":12})

plt.title('Lifestage Distribution of Mainstream Customers',{'fontsize': 15})

behaviour_df.groupby("PREMIUM_CUSTOMER").LIFESTAGE.value_counts()['Mainstream'].plot(kind='bar',fontsize=12)



plt.subplot(133)

plt.ylabel("Number of Transactions",{"fontsize":12})

plt.title('Lifestage Distribution of Premium Customers',{'fontsize': 15})

behaviour_df.groupby("PREMIUM_CUSTOMER").LIFESTAGE.value_counts()['Premium'].plot(kind='bar',fontsize=12)
combined_data = transaction_df.join(behaviour_df.set_index('LYLTY_CARD_NBR'), on = 'LYLTY_CARD_NBR')

combined_data.head()
customer_groups = combined_data.groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).agg({'TOT_SALES':'sum','PROD_QTY':'sum'}).reset_index().sort_values('TOT_SALES')

customer_groups['SEGMENT'] = customer_groups.LIFESTAGE + '_' + customer_groups.PREMIUM_CUSTOMER
x = list(customer_groups.SEGMENT)

y = list(customer_groups.TOT_SALES)

plt.figure(figsize=(15,8))

plt.xlabel("Total Sales",{'fontsize':15})

plt.barh(x,y)
customer_groups = customer_groups.sort_values("PROD_QTY")

x = list(customer_groups.SEGMENT)

y = list(customer_groups.PROD_QTY)

plt.figure(figsize=(15,8))

plt.xlabel("Number of Products Purchased",{'fontsize':15})

plt.barh(x,y)
sales_pc = combined_data.groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).agg({'PROD_QTY':'sum','TOT_SALES':'sum','TXN_ID':'count'}).reset_index().sort_values('TOT_SALES')

sales_pc['SEGMENT'] = customer_groups.LIFESTAGE + '_' + customer_groups.PREMIUM_CUSTOMER

sales_pc['SALES_PC'] = sales_pc.TOT_SALES / sales_pc.TXN_ID

sales_pc['QTY_PC'] = sales_pc.PROD_QTY / sales_pc.TXN_ID

sales_pc['AVG_PP'] = sales_pc.TOT_SALES / sales_pc.PROD_QTY
sales_pc = sales_pc.sort_values("SALES_PC")

x = list(sales_pc.SEGMENT)

y = list(sales_pc.SALES_PC)

plt.figure(figsize=(15,8))

plt.xlabel("Total Sales Per Customer",{'fontsize':15})

plt.barh(x,y)
sales_pc = sales_pc.sort_values("PROD_QTY")

x = list(sales_pc.SEGMENT)

y = list(sales_pc.PROD_QTY)

plt.figure(figsize=(15,8))

plt.xlabel("Total Products purchased Per Customer",{'fontsize':15})

plt.barh(x,y)
sales_pc = sales_pc.sort_values("AVG_PP")

x = list(sales_pc.SEGMENT)

y = list(sales_pc.AVG_PP)

plt.figure(figsize=(15,8))

plt.xlabel("Average Price paid per Product",{'fontsize':15})

plt.barh(x,y)
combined_data["AVG_PACKET"] = combined_data["TOT_SALES"] / combined_data["PROD_QTY"]



data1 = list(combined_data[(combined_data['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES", "MIDAGE SINGLES/COUPLES"]))  & (combined_data['PREMIUM_CUSTOMER'] == 'Mainstream')]["AVG_PACKET"])

data2 = list(combined_data[(combined_data['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES", "MIDAGE SINGLES/COUPLES"]))  & (combined_data['PREMIUM_CUSTOMER'] != 'Mainstream')]["AVG_PACKET"])
from scipy.stats import ttest_ind



stat, p = ttest_ind(data1, data2,equal_var=True)

print('t=%.3f, p=%.3f ' % (stat, p))
import mlxtend

from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules
basket_brand = (combined_data[(combined_data['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (combined_data['PREMIUM_CUSTOMER']=='Mainstream')]

        .groupby(['LYLTY_CARD_NBR','BRAND'])['PROD_QTY']

        .sum().unstack().reset_index().fillna(0)

        .set_index('LYLTY_CARD_NBR'))



def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1



basket_brand = basket_brand.applymap(encode_units)

basket_brand
frequent_itemsets = apriori(basket_brand, min_support=0.07, use_colnames=True)

frequent_itemsets.head()
rules = association_rules(frequent_itemsets, metric="lift")

rules.head()
basket_packet_size= (combined_data.groupby(['LYLTY_CARD_NBR', 'PACKET_SIZE'])['PROD_QTY']

                     .sum().unstack().reset_index().fillna(0).set_index('LYLTY_CARD_NBR'))



def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1



basket_packet_size = basket_packet_size.applymap(encode_units)

basket_packet_size
frequent_itemsets = apriori(basket_packet_size, min_support=0.07, use_colnames=True)

frequent_itemsets.head()
rules = association_rules(frequent_itemsets, metric="lift")

rules.head()
segment1 = (combined_data[(combined_data['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES"]))  

                          & (combined_data['PREMIUM_CUSTOMER'] == 'Mainstream')])

other = (combined_data[~((combined_data['LIFESTAGE'].isin(["YOUNG SINGLES/COUPLES"]))  

                       & (combined_data['PREMIUM_CUSTOMER'] == 'Mainstream'))])
quantity_segment1 =  segment1.groupby(['BRAND'])[['PROD_QTY']].sum().reset_index().set_index("BRAND")

quantity_other = other.groupby(['BRAND'])[['PROD_QTY']].sum().reset_index().set_index("BRAND")



quantity_segment1_by_brand = quantity_segment1.PROD_QTY / segment1.PROD_QTY.sum()

quantity_segment1_by_brand.name = "TARGETTED_SEGMENT"



quantity_other_by_brand = quantity_other.PROD_QTY / other.PROD_QTY.sum()

quantity_other_by_brand.name = "OTHER_SEGMENT"



brand_proportions = pd.concat([quantity_segment1_by_brand,quantity_other_by_brand], names=["TARGETTED_SEGMENT","OTHER_SEGMENT"],axis=1)



brand_proportions["AFFINITY_TO_BRAND"] = brand_proportions.TARGETTED_SEGMENT / brand_proportions.OTHER_SEGMENT

brand_proportions = brand_proportions.sort_values("AFFINITY_TO_BRAND")
brand_proportions.tail()
quantity_segment2 =  segment1.groupby(['PACKET_SIZE'])[['PROD_QTY']].sum().reset_index().set_index("PACKET_SIZE")

quantity_other = other.groupby(['PACKET_SIZE'])[['PROD_QTY']].sum().reset_index().set_index("PACKET_SIZE")



quantity_segment2_by_pack = quantity_segment2.PROD_QTY / segment1.PROD_QTY.sum()

quantity_segment2_by_pack.name = "TARGETTED_SEGMENT"



quantity_other_by_pack = quantity_other.PROD_QTY / other.PROD_QTY.sum()

quantity_other_by_pack.name = "OTHER_SEGMENT"



pack_proportions = pd.concat([quantity_segment2_by_pack,quantity_other_by_pack], axis=1)



pack_proportions["AFFINITY_TO_BRAND"] = pack_proportions.TARGETTED_SEGMENT / pack_proportions.OTHER_SEGMENT

pack_proportions = pack_proportions.sort_values("AFFINITY_TO_BRAND")
pack_proportions.tail()
fig = plt.figure(figsize=(18,10))



#Plotting Affinty by Brand

plt.subplot(121)

my_range = range(1, len(brand_proportions.index) + 1)

color_code = np.where(brand_proportions["AFFINITY_TO_BRAND"]==brand_proportions["AFFINITY_TO_BRAND"].max(), '#ff7f0e', '#1f77b4')

plt.hlines(y=my_range, xmin=0, xmax=brand_proportions['AFFINITY_TO_BRAND'],color=color_code)

plt.scatter(brand_proportions['AFFINITY_TO_BRAND'], my_range, color=color_code)

plt.yticks(np.arange(1,len(brand_proportions)+1), brand_proportions.index)



#Plotting Affinty by Packet Size

plt.subplot(122)

my_range = range(1, len(pack_proportions.index) + 1)

color_code = np.where(pack_proportions["AFFINITY_TO_BRAND"]==pack_proportions["AFFINITY_TO_BRAND"].max(), '#ff7f0e', '#1f77b4')

plt.hlines(y=my_range, xmin=0, xmax=pack_proportions['AFFINITY_TO_BRAND'],color=color_code)

plt.scatter(pack_proportions['AFFINITY_TO_BRAND'], my_range, color=color_code)

plt.yticks(np.arange(1,len(pack_proportions)+1), pack_proportions.index)

plt.show()
combined_data[combined_data["PACKET_SIZE"] == 270].BRAND.unique()
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:,.2f}'.format
df_general = pd.read_csv("../input/olist_public_dataset.csv")
df_general.columns
# Count order_aproved_at
df_general.groupby("order_aproved_at").size().reset_index(name='count').sort_values("count", ascending=False)
df_general[df_general["order_aproved_at"] == "2017-01-27 03:05:27.454387"]\
[["order_aproved_at", "order_delivered_customer_date"]]
# Now I will remove the duplicate orders

df_general.drop_duplicates(subset="order_aproved_at", inplace=True)
df_general.count()
# Select columns

df = df_general[["order_purchase_timestamp", 
         "order_estimated_delivery_date",
         "order_delivered_customer_date",
         "customer_state"]]

df.head()
# Removing rows with null values

df.dropna(inplace=True)
df.info()
df['date_purchase'] = pd.to_datetime(df["order_purchase_timestamp"])
df['date_estimated'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['date_delivered'] = pd.to_datetime(df['order_delivered_customer_date'])
df.info()
# days to be delivered
df['delta_purch_delivered'] = (df['date_delivered'] - df['date_purchase']).dt.days
# estimated delivery days
df['delta_est_delivered'] = (df['date_estimated'] - df['date_purchase']).dt.days

df.head()
# Histogram - estimated delivery days
df['delta_est_delivered'].plot.hist(grid=True, bins=df['delta_est_delivered'].max(), figsize = (8,8))
plt.title('Histogram - days estimated to deliver')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.grid(axis='y')
avg = df['delta_est_delivered'].mean()
std = df['delta_est_delivered'].std()
plt.text(100, 6000, r'$\mu={:,.2f}, \sigma={:,.2f}$'.format(avg, std))
plt.show()

# Histogram - days to deliver
df['delta_purch_delivered'].plot.hist(grid=True, bins=df['delta_purch_delivered'].max(), color='#ff0000', figsize = (8, 8))
plt.title('Histogram - days to deliver')
plt.xlabel('Days')
plt.ylabel('Cases')
avg = df['delta_purch_delivered'].mean()
std = df['delta_purch_delivered'].std()
plt.text(150, 6000, r'$\mu={:,.2f}, \sigma={:,.2f}$'.format(avg, std))
plt.grid(axis='y')
plt.show()
df.boxplot(column=['delta_est_delivered', 'delta_purch_delivered'], by='customer_state', figsize = (25,10), showfliers=False)
plt.show()
# zoom in RJ
df[df["customer_state"] == "RJ"].boxplot(column=['delta_est_delivered', 'delta_purch_delivered'], by='customer_state', figsize = (25,10), showfliers=False)
plt.show()
df["delayed"] = (df['date_estimated'] - df['date_delivered']).dt.days

df_count_delay = df[df["delayed"] < 0].groupby(['customer_state'], as_index=False)["delayed"].count()
df_total = df.groupby(['customer_state'], as_index=False)["delayed"].count()

df_total = df_count_delay.merge(df_total, on='customer_state')
df_total.rename(columns={'delayed_y': 'total_orders'}, inplace=True)

df_total["%"] = (df_total["delayed_x"] / df_total["total_orders"]) * 100
df_total.sort_values(by="%", ascending=False)
# Statistics measures

df_state = df[["delta_purch_delivered", "delta_est_delivered", "customer_state"]]
df_measures = df_state.groupby(['customer_state'], as_index=False).agg(['mean','std', 'var'])

df_measures.head()
# The value in 2 standard deviations
df_measures['2std'] = df_measures["delta_purch_delivered"]["mean"] + (df_measures["delta_purch_delivered"]["std"]*2)

# The difference between the 2 standard deviations and the mean of estimated days delivered
df_measures['delta_std_mean_estimated'] = df_measures['2std'] -  df_measures["delta_est_delivered"]["mean"]
df_measures.sort_values("delta_std_mean_estimated", ascending=False)
df_merge = df_measures["2std"].reset_index(level=0) \
.merge(df_measures["delta_std_mean_estimated"].reset_index(level=0), on="customer_state")\
.merge(df_total, on="customer_state").sort_values("%", ascending=False)

df_merge
# Show the behavior of delayed orders in each state

fig, ax = plt.subplots(figsize=(15,7))
df[(df["delayed"] < 0) & (df["customer_state"] == "RJ")].groupby([df.date_purchase.dt.year, df.date_purchase.dt.month])["delayed"].count().plot(ax=ax)
plt.title('Quantity orders delayed - RJ')
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
df[(df["delayed"] < 0) & (df["customer_state"] == "MA")].groupby([df.date_purchase.dt.year, df.date_purchase.dt.month])["delayed"].count().plot(ax=ax)
plt.title('Quantity orders delayed - MA')
plt.show()

fig, ax = plt.subplots(figsize=(15,7))
df[(df["delayed"] < 0) & (df["customer_state"] == "BA")].groupby([df.date_purchase.dt.year, df.date_purchase.dt.month])["delayed"].count().plot(ax=ax)
plt.title('Quantity orders delayed - BA')
plt.show()


# I will add others variables to create a new dataset 

df = df_general[["order_purchase_timestamp", 
         "order_estimated_delivery_date",
         "order_delivered_customer_date",
         "customer_state",
         "review_comment_message",
        "review_score",
        "product_category_name",
        "order_freight_value"]]

df.dropna(inplace=True)

df['date_purchase'] = pd.to_datetime(df["order_purchase_timestamp"])
df['date_estimated'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['date_delivered'] = pd.to_datetime(df['order_delivered_customer_date'])

df = df[(df["customer_state"] == "RJ") | \
         (df["customer_state"] == "MA") | \
         (df["customer_state"] == "BA")]

df['delta_purch_delivered'] = (df['date_delivered'] - df['date_purchase']).dt.days
df['delta_est_delivered'] = (df['date_estimated'] - df['date_purchase']).dt.days
df["delayed"] = (df['date_estimated'] - df['date_delivered']).dt.days 

df.head()
serie_march18 = df[(df["date_delivered"] >= datetime.date(2018,3,1)) \
                & (df["date_delivered"] <= datetime.date(2018,3,31))]["review_score"]

serie_march17 = df[(df["date_delivered"] >= datetime.date(2017,3,1)) \
                & (df["date_delivered"] <= datetime.date(2017,3,31))]["review_score"]
plt.title('Histogram - Review Score March 2017')
serie_march17.plot.hist(grid=True, figsize = (8,8))
plt.xlabel('Review Score')
avg = serie_march17.mean()
std = serie_march17.std()
plt.text(3, 150, r'$\mu={:,.2f},\ \sigma={:,.2f}$'.format(avg, std))
plt.show()

plt.title('Histogram - Review Score March 2018')
serie_march18.plot.hist(grid=True, figsize = (8,8))
plt.xlabel('Review Score')
avg = serie_march18.mean()
std = serie_march18.std()
plt.text(3, 150, r'$\mu={:,.2f},\ \sigma={:,.2f}$'.format(avg, std))
plt.show()
df.loc[df.customer_state == 'RJ', '2std'] = df["date_purchase"]+datetime.timedelta(38)
df.loc[df.customer_state == 'MA', '2std'] = df["date_purchase"]+datetime.timedelta(41)
df.loc[df.customer_state == 'BA', '2std'] = df["date_purchase"]+datetime.timedelta(42)

df["2std"] = pd.to_datetime(df["2std"])
df["delayed_estimated"] = (df['2std'] - df['date_delivered']).dt.days
# Amount of delayed orders with real estimate

df[df["delayed"] < 0]\
                .groupby(['customer_state'], as_index=False)["delayed"]\
                .count().sort_values(by="delayed", ascending=False)
# Amount of delayed orders with new estimate

df[df["delayed_estimated"] < 0]\
                .groupby(['customer_state'], as_index=False)["delayed_estimated"]\
                .count().sort_values(by="delayed_estimated", ascending=False)
# Comparing

fig, ax = plt.subplots(figsize=(15,7))
df[(df["delayed_estimated"] < 0) & (df["customer_state"] == "RJ")].groupby([df.date_purchase.dt.year, df.date_purchase.dt.month])["delayed_estimated"].count().plot(ax=ax, linestyle='--', label="Estimated 2std")
df[(df["delayed"] < 0) & (df["customer_state"] == "RJ")].groupby([df.date_purchase.dt.year, df.date_purchase.dt.month])["delayed"].count().plot(ax=ax, label="Real")
plt.title('Quantity orders delayed - RJ')
plt.legend()

plt.show()
# Amount of delayed orders with real estimate in march/2018

df[(df["delayed"] < 0) & (df["date_purchase"] >= datetime.date(2018,3,1)) \
                & (df["date_purchase"] <= datetime.date(2018,3,31))] \
                .groupby(['customer_state'], as_index=False)["delayed"]\
                .count().sort_values(by="delayed", ascending=False)
# Amount of delayed orders with new estimate in march/2018

df[(df["delayed_estimated"] < 0) & (df["date_purchase"] >= datetime.date(2018,3,1)) \
                & (df["date_purchase"] <= datetime.date(2018,3,31))] \
                .groupby(['customer_state'], as_index=False)["delayed_estimated"]\
                .count().sort_values(by="delayed_estimated", ascending=False)
text = "Reduced in march 2018 {:,.2f}% in {}"

print(text.format(((201-116)/201)*100, "RJ"))
print(text.format(((56-10)/56)*100, "BA"))
print(text.format(((19-4)/19)*100, "MA"))
df[(df["delayed"] < 0) & (df["date_purchase"] >= datetime.date(2018,3,1)) \
                & (df["date_purchase"] <= datetime.date(2018,3,31))] \
                .groupby(['customer_state'], as_index=False)["order_freight_value"]\
                .sum().sort_values(by="order_freight_value", ascending=False)
# Appling new estimate method

df[(df["delayed_estimated"] < 0) & (df["date_purchase"] >= datetime.date(2018,3,1)) \
                & (df["date_purchase"] <= datetime.date(2018,3,31))] \
                .groupby(['customer_state'], as_index=False)["order_freight_value"]\
                .sum().sort_values(by="order_freight_value", ascending=False)
print("The company would not have lost R$ {} in RJ, MA and BA".format((5029 + 1901 + 681) - (3147+301+184)))

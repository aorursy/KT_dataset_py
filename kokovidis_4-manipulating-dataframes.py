import pandas as pd
df_master = pd.read_csv('../input/orders.csv')
df= df_master.copy()
df.head(10)
statement = df.order_hour_of_day == 8
df8= df[statement]
df8.head()
df8= df[df.order_hour_of_day == 8]
df8.head()
eight_rows = df.order_hour_of_day == 8
eight_rows.head(10)
df.loc[eight_rows, 'order_hour_of_day'] = 9
df.head()
df_clean = df.dropna(how='any')
df_clean.head()
df_clean['days_since_prior_order_int'] = df_clean.days_since_prior_order.apply(int)
# do not worry about the warning message for the time being_
df_clean.head()
del df_clean['days_since_prior_order']
df_clean.head()
df.head()
df[df.order_hour_of_day == 1].order_hour_of_day.count()
#create an empty DataFrame
df_groups = pd.DataFrame()
#groupby "order_hour_of_day" & find the size of the groups (count)
df_groups = df.groupby("order_hour_of_day").count()
df_groups.head()
hour_new_customers =  df_groups.user_id - df_groups.days_since_prior_order 
# sort.values(ascending=False) sorts the values in a descending order.
hour_new_customers.sort_values(ascending=False)

# all orders = 100%
# new orders = ??
pct_hour_new_customers= (hour_new_customers*100)/df_groups.user_id
pct_hour_new_customers.sort_values(ascending=False)
df_groups.user_id.sum() - df_groups.days_since_prior_order.sum() 
df.user_id.nunique()
df_groups_2 = df.groupby(["order_dow", "order_hour_of_day"]).agg("count")
df_groups_2
#even if it is not mandatory, we chain here .to_frame() to convert the series (single column) to a DataFrame

day_hour_order = df_groups_2.order_id.sort_values(ascending=False).to_frame()
day_hour_order
df_groups_2_reset = df_groups_2.reset_index()
df_groups_2_reset
df_groups_2_reset = df_groups_2_reset.loc[:,['order_dow', 'order_hour_of_day', 'order_id']]
df_groups_2_reset.head()
df_groups2_pivot = df_groups_2_reset.pivot(index='order_dow', columns='order_hour_of_day', values='order_id')
df_groups2_pivot
df_groups2_pivot = df_groups_2_reset.pivot(index='order_hour_of_day', columns='order_dow', values= 'order_id')
df_groups2_pivot
df.head(20)
#the aggregation function in our example is the count
df_pivot_table = df.pivot_table(index='order_hour_of_day', columns='order_dow', values= 'order_id' , aggfunc='count')
df_pivot_table
df_pivot_table = df.pivot_table(index='order_hour_of_day', columns='order_dow', values= 'order_id' , aggfunc='count', margins=True)
df_pivot_table
df.shape

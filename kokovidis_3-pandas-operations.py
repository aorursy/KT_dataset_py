import pandas as pd
df = pd.read_csv('../input/orders.csv', sep=',')
df
df.head(7)
df.tail(7)
df.info()
type(df.eval_set)
df.order_dow.value_counts()
df.value_counts()
df.order_dow.value_counts().sort_index()
s1=df['order_id']
type(s1)
s1.head()
dcol1=df[['order_id']]
type(dcol1)
df.order_id
type(df.order_id)
df.order_id.head()
dcol2 = df[['order_id', 'order_dow', 'order_hour_of_day' ]]
dcol2.head()
drow1 = df[6:13]
drow1
dfloc= df.loc[ 4:9 ,['order_id','order_hour_of_day']]
dfloc
dfiloc= df.iloc[ 4:9 ,[1,6]]
dfiloc
dfiloc2= df.iloc[ [4,9] , 1:6 ]
dfiloc2
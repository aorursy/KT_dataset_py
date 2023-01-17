import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
%matplotlib inline
try:
    conn = psycopg2.connect( host='rc1c-fhrb9f1e0l9g611h.mdb.yandexcloud.net',
            port=6432,
            dbname='hr-analytics',
            sslmode='require',
            user='analytics',
            password='HRanalytics')
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("You are connected to - ", record)
except (Exception, psycopg2.Error) as error:
    print ("Error while connecting to PostgreSQL", error)
orders_df = pd.read_sql("SELECT ord.Id AS order_id, dw.store_id, dw.starts_at AS starts_delivery, dw.ends_at AS ends_delivery, \
                         ord.shipped_at AS time_delivery, ord.state AS state_delivery \
                         FROM orders ord \
                         JOIN delivery_windows dw ON ord.delivery_window_id = dw.id", conn)


#orders_PR = ProfileReport(orders_df, title='EDA', explorative=True)
#orders_PR
orders_df = orders_df.drop_duplicates(subset=['order_id']).reset_index(drop=True)
orders_df['state_delivery'].unique()
orders_df = pd.concat([orders_df, pd.get_dummies(orders_df['state_delivery'])], axis=1).reset_index(drop=True)
orders_df
orders_df.info()
all_days_delivery = orders_df.set_index('time_delivery').groupby(pd.Grouper(freq='D')).sum()
all_days_delivery.drop(['order_id'], axis = 1, inplace=True)
all_days_delivery 
all_month_delivery = all_days_delivery.groupby(pd.Grouper(freq='M')).sum()
all_month_delivery 
orders_df['delay'] = orders_df['time_delivery'].sub(orders_df['ends_delivery'], axis=0)
orders_day = orders_df.loc[orders_df['time_delivery'] >= '2019-08-04'].reset_index(drop=True) 
orders_day







'''cursor.close()
conn.close()'''
import pandas as pd



df= pd.read_csv('/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv')

df['event_time'] = pd.to_datetime(df['event_time'])

df['day-hr'] = df['event_time'].dt.strftime('%d/%m/%Y-%H')
def get_brushers(shopid):

    

    by_id = df[df['shopid']==shopid].groupby(['day-hr','userid']).agg({'orderid':'count'}).reset_index()

    brusher_ids = by_id[by_id['orderid'] >=3]['userid'].sort_values().astype(str)

    if len(brusher_ids):

        return '&'.join(brusher_ids)

    else:

        return 0
df_submit = pd.DataFrame()

df_submit['shopid'] = df['shopid'].unique()

df_submit['userid'] = df_submit['shopid'].apply(get_brushers)

df_submit.to_csv('submission.csv', index=False)
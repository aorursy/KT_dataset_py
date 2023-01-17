import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings('ignore')



import os
brand_shoptype = pd.read_csv('/kaggle/input/ptr-rd1/Extra_material_2.csv')

transaction = pd.read_csv('/kaggle/input/ptr-rd1/Extra_material_3.csv', parse_dates=['date_id'])



print('brand_shoptype shape:', brand_shoptype.shape)

print('transaction shape:', transaction.shape)
all_brands = np.sort(brand_shoptype.brand.unique())
# filter only official shop

brand_shoptype = brand_shoptype[brand_shoptype['shop_type'] == 'Official Shop']



brand_shoptype['shop_id'] = brand_shoptype['shop_id'].astype(int)
# filter only transaction from 10th May to 31st May

start_date = '2019-5-10'

end_date = '2019-5-31'



mask = (transaction['date_id'] >= start_date) & (transaction['date_id'] <= end_date)



transaction = transaction.loc[mask]
# drop duplicates row

transaction.drop_duplicates(inplace=True)

brand_shoptype.drop_duplicates(inplace=True)
error_idx = brand_shoptype[brand_shoptype['brand'] == 'Anessa, Senka, Za, Tsubaki, Ma Cherie'].index



brand_shoptype.drop(error_idx, inplace=True)
# filter only shop_id which is exist in transaction

brand_shoptype = brand_shoptype[brand_shoptype['shop_id'].isin(transaction['shopid'])]
brand_shoptype.head()
transaction.head()
# there are 4 shop_id that have 2 brands

brand_shoptype.groupby('shop_id')['brand'].nunique().sort_values(ascending=False)[:5]
transaction.rename(columns={'shopid': 'shop_id'}, inplace=True)



final_df = brand_shoptype.merge(transaction, on='shop_id', how='left')



final_df.head()
final_df['gross_sales_revenue'] = final_df['amount'].values * final_df['item_price_usd'].values
final_df.sample(5)
for_submission = pd.DataFrame({})



for_submission['Index'] = range(1, len(all_brands) + 1)

for_submission['Answers'] = ''



for idx, brand in enumerate(all_brands):

    result = final_df[final_df['brand'] == brand].groupby('itemid')['gross_sales_revenue'].sum().rename('gross_sales').sort_values(ascending=False)[:3].reset_index()

    if len(result['itemid']) > 0:

        for_submission.at[idx, 'Answers'] += (str(brand) + ', ')

        for itemid in result['itemid']:

            for_submission.at[idx, 'Answers'] += (str(itemid) + ', ')

    else:

        for_submission.at[idx, 'Answers'] += (str(brand) + ', N.A,,')

        

for_submission['Answers'] = for_submission['Answers'].apply(lambda x: x[:-2])
for_submission.to_csv('submission1.csv', index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



items = pd.read_csv(r'../input/competitive-data-science-predict-future-sales/items.csv')

items_category = pd.read_csv(r'../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv(r'../input/competitive-data-science-predict-future-sales/shops.csv')

sales_train = pd.read_csv(r'../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv(r'../input/competitive-data-science-predict-future-sales/test.csv')





sales_train.drop(['date'],axis =1, inplace=True)

sales_train['data_origin'] = 'train'



test['data_origin'] = 'test'

test['date_block_num'] = 34

items_category['item_category_name2']= items_category['item_category_name'].str.split('-').apply(lambda x : x[0])

shops['shop_city'] = shops['shop_name'].str.split(" ").apply(lambda x : x[0])



medium_prices_for_shops = sales_train.groupby(['shop_id','item_id'], as_index = False)['item_price'].mean()

test = pd.merge(test, medium_prices_for_shops, how = 'left', on = ['shop_id','item_id'])



items = pd.merge(items,items_category,how ='left',on ='item_category_id')



item_medium_price = sales_train.groupby(['item_id'],as_index = False)['item_price'].mean()



items= pd.merge(items,item_medium_price,on ='item_id',how= 'left')

items_category_name2_medium_price = items.groupby(['item_category_name2'],as_index=False)['item_price'].mean()

items = pd.merge(items, items_category_name2_medium_price, how= 'left', on = 'item_category_name2')



items.rename(columns = {'item_price_x':'item_medium_price', 'item_price_y':'item_medium_price_category2'}, inplace = True)

test['item_price'].fillna(value = items['item_medium_price_category2'],inplace = True)



test['item_cnt_day'] = 0





# grouping data at the level of month for sales_train



sales_train_aggregated = sales_train.groupby(['date_block_num','shop_id','item_id','item_price','data_origin'],as_index= False)['item_cnt_day'].sum()



#test.drop(columns ='ID',axis =1, inplace = True)



data_concatenated  = sales_train_aggregated.append(test, ignore_index = True, sort = False)



data_concatenated.sort_values(by = ['shop_id','item_id','date_block_num'], ascending = True, inplace = True)



data_concatenated['diff1'] = data_concatenated['item_cnt_day'].diff(periods =1)

mask = data_concatenated.item_id != data_concatenated.item_id.shift(1)

data_concatenated['diff1'][mask] = np.nan



data_concatenated['diff2'] = data_concatenated['item_cnt_day'].diff(periods =2)

mask = data_concatenated.item_id != data_concatenated.item_id.shift(1)

data_concatenated['diff2'][mask] = np.nan



data_concatenated['diff3'] = data_concatenated['item_cnt_day'].diff(periods =3)

mask = data_concatenated.item_id != data_concatenated.item_id.shift(1)

data_concatenated['diff3'][mask] = np.nan



data_concatenated['diff4'] = data_concatenated['item_cnt_day'].diff(periods =4)

mask = data_concatenated.item_id != data_concatenated.item_id.shift(1)

data_concatenated['diff4'][mask] = np.nan



data_concatenated['diff5'] = data_concatenated['item_cnt_day'].diff(periods =5)

mask = data_concatenated.item_id != data_concatenated.item_id.shift(1)

data_concatenated['diff5'][mask] = np.nan



data_concatenated['diff6'] = data_concatenated['item_cnt_day'].diff(periods =6)

mask = data_concatenated.item_id != data_concatenated.item_id.shift(1)

data_concatenated['diff6'][mask] = np.nan



data_concatenated.fillna(value = 0, inplace = True)



data_concatenated.merge(items[['item_id','item_category_id']], on = 'item_id' , how = 'left')



from sklearn.linear_model import LinearRegression





y_train  = data_concatenated[data_concatenated['data_origin'] == 'train']['item_cnt_day']

x_col  =  [col for col in data_concatenated.columns if col not in ['data_origin','item_cnt_day','ID']] 





x_train = data_concatenated[data_concatenated['data_origin'] == 'train'][x_col]

reg = LinearRegression().fit(x_train,y_train)



x_test = data_concatenated[data_concatenated['data_origin'] == 'test'][x_col]

y_pred = reg.predict(x_test)





my_submission = pd.DataFrame({'ID':test.ID, 'item_cnt_month':y_pred})



my_submission.to_csv('submission.csv', index=False)



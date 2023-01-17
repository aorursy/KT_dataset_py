

import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
train_data = pd.read_csv('../input/sales_train.csv')
test_data = pd.read_csv('../input/test.csv')

train_data.columns
dataset = train_data.pivot_table(index = ['shop_id','item_id'],columns = ['date_block_num'],values = ['item_cnt_day'],fill_value = 0)


dataset.reset_index(inplace = True)
dataset = pd.merge(test_data,dataset,how = 'left',on = ['shop_id','item_id'])
dataset.head()
dataset.fillna(0,inplace = True)
dataset.head()
submission_pfs = dataset.iloc[:,36]
submission_pfs.clip(0,20,inplace = True)
submission_pfs = pd.DataFrame({'ID':test_data['ID'],'item_cnt_month':submission_pfs.ravel()})

submission_pfs.head()
submission_pfs.to_csv('pfs.csv',index = False)
g = pd.read_csv('pfs.csv')
g.head()

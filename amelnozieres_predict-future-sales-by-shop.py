import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline 
# Import data
transactions  = pd.read_csv("../input/sales_train.csv")

item_categories =pd.read_csv("../input/item_categories.csv")
items = pd.read_csv("../input/items.csv")
subm=pd.read_csv("../input/sample_submission.csv")
shops=pd.read_csv("../input/shops.csv")
test=pd.read_csv("../input/test.csv")



print(transactions.shape)
print(transactions.head(5))
print(items.shape)
print(items.head(5))
print(item_categories.shape)
print(item_categories.head(5))
print(shops.shape)
print(shops.head(5))
transactions['date'] = pd.to_datetime(transactions['date'],format = '%d.%m.%Y')
print(transactions.head(5))
print(transactions.tail(5))
print(test.head())
submission = test[['ID']]

submission['item_cnt_month'] = 0.188
print(submission.head(5))
submission.to_csv('submission_01.csv',index= False)



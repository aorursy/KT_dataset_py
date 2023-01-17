import numpy as np
import pandas as pd 
import os
print(os.listdir("../input/"))
train = pd.read_csv('../input/train_items.csv.zip', compression='zip')
train.head()
train.isnull().sum()
apriori_med = train.groupby(['item_condition_id', 
                             'shipping', 
                             'category_name'])['price'].agg('median').to_frame().reset_index()
apriori_med.head()

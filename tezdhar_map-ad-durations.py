import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv", usecols=['item_id', 'user_id', 'deal_probability'], 
                    dtype={'item_id': str, 'user_id': str, 'deal_probability': float})
train.head()
periods_train = pd.read_csv("../input/periods_train.csv", parse_dates = ['activation_date', 'date_from', 'date_to'],
                    dtype={'item_id': str})
periods_train.head()
#Check common item id's
print(len(set(train.item_id)))
print(len(set(periods_train.item_id)))
print(len(set(train.item_id).intersection(set(periods_train.item_id))))
train_active = pd.read_csv("../input/train_active.csv", usecols=["item_id", "user_id", "item_seq_number"])
train_active.head()
print(len(set(train_active.item_id)))
print(len(set(train.item_id).intersection(set(train_active.item_id))))

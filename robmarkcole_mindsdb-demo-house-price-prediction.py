import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
train_df.head()
from mindsdb import *

# First we initiate MindsDB
mdb = MindsDB()
# We tell mindsDB what we want to learn and from what data
mdb.learn(
    from_file=train_df, # 
    predict='SalePrice', 
    model_name='kaggle_house_sale'
)
# Here we use the model to make predictions (NOTE: You need to run train.py first)
result = mdb.predict(predict='SalePrice', when={"MSSubClass": 20, "MSZoning": "Rh","LotFrontage":80,"LotArea":11622}, model_name='kaggle_house_sale')

print(result.predicted_values)
# you can now print the results
print('The predicted price is ${price} with {conf} confidence'.format(price=result.predicted_values[0]['SalePrice'], conf=result.predicted_values[0]['prediction_confidence']))

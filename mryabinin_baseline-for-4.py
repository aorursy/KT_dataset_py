import pandas as pd

import numpy as np
import json

items_list=[]

with open('../input/items.json/items.json') as inf:

    for line in inf:

        item=json.loads(line)

        if isinstance(item['image'],float):

            item['image']=[0 for _ in range(96)]

        item['image']=np.array(item['image'])

        items_list.append(item)



items=pd.DataFrame(items_list).set_index('itemId')
num_users=42977

num_items=len(items)
import scipy.sparse
from tqdm import tqdm_notebook
data=[]

row=[]

col=[]

with open('../input/train.json/train.json') as inf:

    for i,line in enumerate(tqdm_notebook(inf)):

        j = json.loads(line)

        for item, rating in j["trainRatings"].items():

            data.append((-1)**(int(rating)+1))

            row.append(i)

            col.append(int(item))

train_int=scipy.sparse.coo_matrix((data,(row,col)))
import lightfm

model=lightfm.LightFM(no_components=50,loss='logistic',random_state=0)

model.fit(train_int,epochs=20,num_threads=4)
sample=pd.read_csv('../input/random_benchmark.csv')

sample['pred']=model.predict(sample.userId.values,sample.itemId.values,num_threads=20)

sample.sort_values(["userId", "pred"], ascending=[True, False], inplace=True)

sample.drop(columns=['pred'],inplace=True)

sample.to_csv('lightfm_no_content.csv', index=False)
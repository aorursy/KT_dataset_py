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

from fastai.collab import CollabDataBunch,collab_learner
df = pd.read_json("/kaggle/input/bigdatafinal007/Office_Products_5.json", lines=True)

print(df.columns)

print(df.shape)
df_required = df[['asin', 'reviewerID', 'overall']]

df_required.columns= ['Product_ID','User_ID','Rating']

df_required
#import pandas as pd

#BlackFriday = pd.read_csv("../input/BlackFriday.csv")
#BlackFriday.head()
#Data= BlackFriday[['User_ID','Product_ID','Rating']]

Data=df_required[['User_ID','Product_ID','Rating']]

Data.head()

data= CollabDataBunch.from_df(Data,seed=42,valid_pct=0.2,bs=64)

data.show_batch()
len(Data.Product_ID.unique())
len(data.dl(ds_type='validation'))
import torch.nn.functional as F



learn= collab_learner(data,n_factors=50,loss_func=F.mse_loss,y_range=[0,5],wd=0.1)
learn.lr_find()

learn.recorder.plot()
#learn.fit_one_cycle(5,1e-02)

learn.fit_one_cycle(10,1e-01)
(users,products),ratings = next(iter(data.dl(ds_type='valid')))

preds= learn.model(users,products)
preds
for p in list(zip(preds,ratings)):

    print(p[0].data.cpu().numpy(),'..............' ,p[1].data.cpu().numpy())
list=[]

#for q in Data.User_ID.unique():

for r in Data.Product_ID.unique():

    #list.append(['1005269',r,0])

    list.append(['A234HXDATOAYEY',r,0])

df1= pd.DataFrame(list,columns=['User_ID','Product_ID','Rating'])

data1= CollabDataBunch.from_df(df1,valid_pct=0,bs=2500)



data1.train_dl
print(data)

print(data1)
(users1,products1),ratings1 = next(iter(data1.dl(ds_type='train')))

preds1= learn.model(users1,products1)
df2=pd.DataFrame(preds1.data.cpu().numpy(),columns=['Ratings_new'])

df3=pd.merge(df1,df2,left_index=True, right_index=True)
#df3.info()

df3.sort_values(['Ratings_new'],ascending=False)
preds
#import torch

#list= []

#for q in Data.Product_ID.unique():

#    list.append([1001835,q,0])

#df1= pd.DataFrame(list)

#data1= CollabDataBunch.from_df(df1)

##data1.show_batch()

#preds1=learn.model(data1)

#learn.loss_func

#learn.metrics

#import numpy as np

#np.sqrt(2.05)
#learn.fit_one_cycle(20,1e-6)
learn1 = collab_learner(data, use_nn=True, emb_szs={'User_Id': 10, 'Product_ID':10}, layers=[256, 128, 64], y_range=(1, 5))

learn1.lr_find() # find learning rate

learn1.recorder.plot() # plot learning rate graph
learn1.fit_one_cycle(5,1e-01)
preds1_1= learn1.model(users1,products1)
df2=pd.DataFrame(preds1_1.data.cpu().numpy(),columns=['Ratings_new'])

df3=pd.merge(df1,df2,left_index=True, right_index=True)
#df3.info()
df3.sort_values(['Ratings_new'],ascending=False)
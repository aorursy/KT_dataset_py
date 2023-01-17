# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

    

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

test_df = pd.read_csv('/kaggle/input/gc20202/gc/test.csv')

train_df = pd.read_csv("/kaggle/input/gc20202/gc/train.csv")
from fastai.tabular import *

import pandas as pd

import numpy as np
defaults.device
path_train='/kaggle/input/da-hall/train.csv'

path_test='/kaggle/input/da-hall/test.csv'

train_df.head()
new = train_df["timestamp"].str.split(" ", n = 1, expand = True) 

  

# making separate first name column from new data frame 

train_df["Date"]= new[0] 

  

# making separate last name column from new data frame 

train_df["Time"]= new[1] 





  
train_df["Date"] = pd.to_datetime(train_df["Date"])

train_df.head()
new = train_df["Time"].str.split(":", n = 2, expand = True) 

  

# making separate first name column from new data frame 

train_df["Hour"]= new[0] 

  

# making separate last name column from new data frame 

train_df["Minute"]= new[1]



train_df["Second"]= new[1]
new = test_df["timestamp"].str.split(" ", n = 1, expand = True) 

  

# making separate first name column from new data frame 

test_df["Date"]= new[0] 

  

# making separate last name column from new data frame 

test_df["Time"]= new[1] 



new = test_df["Time"].str.split(":", n = 2, expand = True) 

  

# making separate first name column from new data frame 

test_df["Hour"]= new[0] 

  

# making separate last name column from new data frame 

test_df["Minute"]= new[1]



test_df["Second"]= new[1]
test_df.head()
train_df.head()
add_datepart(train_df,"Date",drop=False)

add_datepart(test_df,"Date",drop=False)

train_df.dtypes
train_df['Hour'] = train_df['Hour'].astype(int)

train_df['Minute'] = train_df['Minute'].astype(int)

train_df['Second'] = train_df['Second'].astype(int)
train_df.dtypes
new1 = (5<=train_df['Hour'])

new2 =  (train_df['Hour']<18)

new = new1&new2

train_df.insert(5, "Open",new, True)
train_df.head(100)
train_df['Is_quarter_start'].head(100)
test_df.head()
len(train_df),len(test_df)
train_df.columns
procs=[FillMissing, Categorify, Normalize]
cat_vars = ['building_number','Open',

       'Month', 'Week', 'Day', 'Dayofweek', 'Is_month_end',

       'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end',

       'Is_year_start'] 

cont_vars = ['Hour','Dayofyear','Minute','Second']

dep_var=['main_meter','sub_meter_1','sub_meter_2']
train_df.head(100)
valid_idx=range(100)
valid_idx
train_df = train_df.drop(columns=['timestamp','Time','Date','Elapsed','Year'])
train_df[dep_var].head()
test_df['Hour'] = test_df['Hour'].astype(int)

test_df['Minute'] = test_df['Minute'].astype(int)

test_df['Second'] = test_df['Second'].astype(int)



new1 = (5<=test_df['Hour'])

new2 =  (test_df['Hour']<18)

new = new1&new2

test_df.insert(5, "Open",new, True)



test_df = test_df.drop(columns=['timestamp','Time','Date','Elapsed','Year'])



test_df.head()
data = (TabularList.from_df(train_df,cat_names=cat_vars,cont_names=cont_vars,procs=procs,)

                .split_by_idx(valid_idx)

                .label_from_df(cols=dep_var, label_cls=FloatList, log=False).add_test(TabularList.from_df(test_df, cat_names=cat_vars))

                .databunch())
data.show_batch()
max_y = np.max(train_df[['main_meter','sub_meter_1','sub_meter_2']])*1.2;

max_y;

min_y = np.min(train_df[['main_meter','sub_meter_1','sub_meter_2']])-500;

min_y
y_range = torch.tensor([[min_y[0],min_y[1],min_y[2]],[max_y[0],max_y[1],max_y[2]]], device=defaults.device)
y_range
learn = tabular_learner(data, layers=[5000,4000,3000,1000,500], ps=[0.01,0.01,0.001,0.001,0.0001],y_range=y_range,emb_drop=0.04, metrics=rmse)
learn.model
learn.lr_find()
learn.recorder.plot()
lr=1e-2
x = learn.fit_one_cycle(10,slice(lr),wd=0.2)
x,y=next(iter(learn.data.valid_dl))
x_pred=learn.model(*x)
y[:30],x_pred[:30]
import matplotlib.pyplot as plt

x1 = x_pred.detach().cpu().numpy()

y1 = y.detach().cpu().numpy()

x1
plt.plot(y1[:,2])

plt.plot(x1[:,2])
test_df.info()
train_df.info()
preds= learn.get_preds(ds_type = DatasetType.Test)[0]
len(preds)
pred_fin = pd.DataFrame(np.array(preds))

pred_fin.to_csv('Final_sub.csv',index=False)
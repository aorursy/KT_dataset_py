import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from tqdm import tqdm

import gc
#============================#

def get_cat(inp):

    tokens = inp.split("_")

    return tokens[0]

#============================#

def get_dept(inp):

    tokens = inp.split("_")

    return tokens[0] + "_" + tokens[1]

#============================#
#Building all the aggregation levels
l12 = pd.read_csv("../input/m5-forecasting-uncertainty/sales_train_evaluation.csv")
l12.head()


l12.id = l12.id.str.replace('_evaluation', '')
l12.head()
COLS = [f"d_{i+1}" for i in range(1941)]
%%time

print("State & Item")

l11 = l12.groupby(['state_id','item_id']).sum().reset_index()

l11["store_id"] = l11["state_id"]

l11["cat_id"] = l11["item_id"].apply(get_cat)

l11["dept_id"] = l11["item_id"].apply(get_dept)

l11["id"] = l11["state_id"] + "_" + l11["item_id"]

print("Item")

l10 = l12.groupby('item_id').sum().reset_index()

l10['id'] = l10['item_id'] + '_X'

l10["cat_id"] = l10["item_id"].apply(get_cat)

l10["dept_id"] = l10["item_id"].apply(get_dept)

l10["store_id"] = 'X'

l10["state_id"] = 'X'

print("Store & Dept")

l9 = l12.groupby(['store_id','dept_id']).sum().reset_index()

l9["cat_id"] = l9["dept_id"].apply(get_cat)

l9["state_id"] = l9["store_id"].apply(get_cat)

l9["item_id"] = l9["dept_id"]

l9["id"] = l9["store_id"] + '_' + l9["dept_id"]

print("Store & Cat")

l8 = l12.groupby(['store_id','cat_id']).sum().reset_index()

l8['dept_id'] = l8['cat_id']

l8['item_id'] = l8['cat_id']

l8['state_id'] = l8['store_id'].apply(get_cat)

l8["id"] = l8["store_id"] + '_' + l8["cat_id"]

print("State & Dept")

l7 = l12.groupby(['state_id','dept_id']).sum().reset_index()

l7["store_id"] = l7["state_id"]

l7["cat_id"] = l7["dept_id"].apply(get_cat)

l7["item_id"] = l7["dept_id"]

l7["id"] = l7["state_id"] + '_' + l7["dept_id"]

print("State & Cat")

l6 = l12.groupby(['state_id','cat_id']).sum().reset_index()

l6["store_id"] = l6["state_id"]

l6["dept_id"] = l6["cat_id"]

l6["item_id"] = l6["cat_id"]

l6["id"] = l6["state_id"] + "_" + l6["cat_id"]

print("Dept")

l5 = l12.groupby('dept_id').sum().reset_index()

l5["cat_id"] = l5["dept_id"].apply(get_cat)

l5["item_id"] = l5["dept_id"]

l5["state_id"] = "X"

l5["store_id"] = "X"

l5["id"] = l5["dept_id"] + "_X"

print("Cat")

l4 = l12.groupby('cat_id').sum().reset_index()

l4["store_id"] = l4["cat_id"]

l4["item_id"] = l4["cat_id"]

l4["store_id"] = "X"

l4["state_id"] = "X"

l4["id"] = l4["cat_id"] + "_X"

print("Store")

l3 = l12.groupby('store_id').sum().reset_index()

l3["state_id"] = l3["store_id"].apply(get_cat)

l3["cat_id"] = "X"

l3["dept_id"] = "X"

l3["item_id"] = "X"

l3["id"] = l3["store_id"] + "_X"

print("State")

l2 = l12.groupby('state_id').sum().reset_index()

l2["store_id"] = l2["state_id"]

l2["cat_id"] = "X"

l2["dept_id"] = "X"

l2["item_id"] = "X"

l2["id"] = l2["state_id"] + "_X"

print("Total")

l1 = l12[COLS].sum(axis=0).values

l1 = pd.DataFrame(l1).T

l1.columns = COLS

l1["id"] = 'Total_X'

l1['state_id'] = 'X'

l1['store_id'] = 'X'

l1['cat_id'] = 'X'

l1['dept_id'] = 'X'

l1['item_id'] = 'X'
l11.head()
df = pd.DataFrame()

df = df.append([l12, l11, l10, l9, l8, l7, l6, l5, l4, l3, l2, l1])
df.shape
sub = pd.read_csv("../input/m5-forecasting-uncertainty/sample_submission.csv")

sub['id'] = sub.id.str.replace('_evaluation', '')

grps =sub.iloc[-42840:, 0].unique()

grps = [col.replace("_0.995","") for col in grps]
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, df[col].nunique())
#Computing scale and start date
X = df[COLS].values
X = df[COLS].values

x = (X>0).cumsum(1)

x = x>0

st = x.argmax(1)

den = 1941 - st - 2

diff = np.abs(X[:,1:] - X[:,:-1])

norm = diff.sum(1) / den
st
df["start"] = st

df["scale"] = norm
df.head(5)
plt.plot(X[-1]/norm[-1])

plt.show()
df.to_csv("sales.csv", index=False)
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l11[col].nunique())
X_State_Item = l11[COLS].values


x1 = (X_State_Item>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_State_Item[:,1:] - X_State_Item[:,:-1])

norm = diff.sum(1) / den
l11["start"] = st

l11["scale"] = norm

l11.head()
l11.to_pickle('State_Item_1.pkl')

#l11.to_csv('State_Item_1.csv')
l11.head()
l11.info()
del l11
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l1[col].nunique())
X_Total = l1[COLS].values
X_Total


x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l1["start"] = st

l1["scale"] = norm

l1.head()
l1.to_pickle('TotalSales.pkl')
del l1
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l10[col].nunique())
X_Total = l10[COLS].values


x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l10["start"] = st

l10["scale"] = norm

l10.head()
l10.to_pickle('Items.pkl')
del l10
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l9[col].nunique())
X_Total = l9[COLS].values


x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l9["start"] = st

l9["scale"] = norm

l9.head()
l9.to_pickle('Store_Dept.pkl')
del l9
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l8[col].nunique())
X_Total = l8[COLS].values


x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l8["start"] = st

l8["scale"] = norm

l8.head()
l8.to_pickle('Store_Cat.pkl')
del l8
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l7[col].nunique())
X_Total = l7[COLS].values


x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l7["start"] = st

l7["scale"] = norm

l7.head()
l7.to_pickle('State_Dept.pkl')
del l7
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l6[col].nunique())
X_Total = l6[COLS].values


x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l6["start"] = st

l6["scale"] = norm

l6.head()
l6.to_pickle('State_Category.pkl')
del l6
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l5[col].nunique())
X_Total = l5[COLS].values

x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l5["start"] = st

l5["scale"] = norm

l5.head()
l5.to_pickle('Department.pkl')

del l5
l4
for col in ['id','item_id','cat_id','store_id','state_id']:

    print(col, l4[col].nunique())
X_Total = l4[COLS].values

x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l4["start"] = st

l4["scale"] = norm

l4.head()
l4.to_pickle('Category.pkl')

del l4
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l3[col].nunique())
X_Total = l3[COLS].values

x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l3["start"] = st

l3["scale"] = norm

l3.head()
l3.to_pickle('Store.pkl')

del l3
for col in ['id','item_id','dept_id','cat_id','store_id','state_id']:

    print(col, l2[col].nunique())
X_Total = l2[COLS].values

x1 = (X_Total>0).cumsum(1)

x1 = x1>0

st = x1.argmax(1)

den = 1941 - st - 2

diff = np.abs(X_Total[:,1:] - X_Total[:,:-1])

norm = diff.sum(1) / den
l2["start"] = st

l2["scale"] = norm

l2.head()
l2.to_pickle('State.pkl')

del l2
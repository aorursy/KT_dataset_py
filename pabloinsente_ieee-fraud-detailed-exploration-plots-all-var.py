import numpy as np 

import pandas as pd 



import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
# load data

train_tran = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

train_iden = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_tran = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

test_iden = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_sub = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
# Join training datasets

train = train_tran.merge(train_iden, how='left',left_index=True, right_index=True)

train.shape
train.head()
# Join testing datasets

test = test_tran.merge(test_iden, how='left',left_index=True, right_index=True)

test.shape
test.head()
# get target feature

y_train = train['isFraud'].copy()

y_train.shape
y_train.head()
# get features matrices

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()
del test, train_tran, train_iden, test_tran, test_iden
f, axes = plt.subplots(1, 3, figsize=(12, 4))

isFraud = sns.countplot(x='isFraud', data=train, ax=axes[0])

ProductCD = sns.countplot(x='ProductCD', data=train, ax=axes[1])

DeviceType = sns.countplot(x='DeviceType', data=train, ax=axes[2])

plt.tight_layout()
# First create a dataframe with 2 cols: device info and the count by device

group = pd.DataFrame()

group['DeviceCount'] = train.groupby(['DeviceInfo'])['DeviceInfo'].count()

group['DeviceInfo'] = group.index



# There are too many Devices, so we will subset the top 20

group_top = group.sort_values(by='DeviceCount',ascending=False).head(20)



plt.figure(figsize=(25, 10))

sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.barplot(x="DeviceInfo", y="DeviceCount", data=group_top)

xt = plt.xticks(rotation=60)
# These cards are encoded as float64, and since there are too many values

# we will plot this a distributions (x a# xis is just an id)

f, axes = plt.subplots(4, 1, figsize=(25, 30))



c1 = sns.distplot(train.card1,kde=False, ax=axes[0])

c2 = sns.distplot(train.card2.dropna(),kde=False, ax=axes[1])

c3 = sns.distplot(train.card3.dropna(),kde=False, ax=axes[2])

c5 = sns.distplot(train.card5.dropna(),kde=False, ax=axes[3])
# Plot IV: cards 4 and 6

f, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.set(color_codes=True)

card4 = sns.countplot(x='card4', data=train, ax=axes[0])

card6 = sns.countplot(x='card6', data=train, ax=axes[1])
# First create a dataframe with 2 cols: device info and the count by device

group = pd.DataFrame()

group['addr1Count'] = train.groupby(['addr1'])['addr1'].count()

group['addr1'] = group.index



# There are too many addr, so we will subset the top 20

group_top = group.sort_values(by='addr1Count',ascending=False).head(20)



plt.figure(figsize=(25, 10))

sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.barplot(x="addr1", y="addr1Count", data=group_top)

xt = plt.xticks(rotation=60)
# First create a dataframe with 2 cols: device info and the count by device

group = pd.DataFrame()

group['addr2Count'] = train.groupby(['addr2'])['addr2'].count()

group['addr2'] = group.index



# There are too many addr, so we will subset the top 20

group_top = group.sort_values(by='addr2Count',ascending=False).head(20)



plt.figure(figsize=(25, 10))

sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.barplot(x="addr2", y="addr2Count", data=group_top)

xt = plt.xticks(rotation=60)
# Let's examine addr2 values

train.addr2.value_counts().head(10)
f, axes = plt.subplots(1, 2, figsize=(18, 12))



sns.set(color_codes=True)

p_email = sns.countplot(y='P_emaildomain', data=train, ax=axes[0])

r_email = sns.countplot(y='R_emaildomain', data=train, ax=axes[1])

plt.tight_layout()
print(train.addr1.nunique()) # 332 unique locations

print(train.addr2.nunique()) # 74 unique locations

print(train.P_emaildomain.nunique()) # 59 unique domains

print(train.R_emaildomain.nunique()) # 59 unique domains
# Plot VIII: M1 - M9 variables

M1_loc = train.columns.get_loc("M1")

M9_loc = train.columns.get_loc("M9")

df_m = train.iloc[:,M1_loc:M9_loc+1] #subset dataframe M1-M9



cols = df_m.columns

f, axes = plt.subplots(3, 3, figsize=(16, 12))

count = 0

for i in range(3): # rows loop

    for j in range(3): # cols loop

        mplot = sns.countplot(x=cols[count], data=df_m, ax=axes[i,j])

        count += 1 # to loop over col-names

plt.tight_layout()
#Exploration id_12 - id_38

id12_loc = train.columns.get_loc("id_12")

id38_loc = train.columns.get_loc("id_38")

df_id = train.iloc[:,id12_loc:id38_loc+1] #subset dataframe id12-id19
df_id.dtypes
df_id.head(15)
# First create a dataframe with 2 cols: device info and the count by device

group = pd.DataFrame()

group['id_30Count'] = df_id.groupby(['id_30'])['id_30'].count()

group['id_30'] = group.index



# There are too many addr, so we will subset the top 20

group_top = group.sort_values(by='id_30Count',ascending=False).head(20)



plt.figure(figsize=(25, 10))

sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.barplot(x="id_30", y="id_30Count", data=group_top)

xt = plt.xticks(rotation=60)
# First create a dataframe with 2 cols: device info and the count by device

group = pd.DataFrame()

group['id_31Count'] = df_id.groupby(['id_31'])['id_31'].count()

group['id_31'] = group.index



# There are too many addr, so we will subset the top 20

group_top = group.sort_values(by='id_31Count',ascending=False).head(20)



plt.figure(figsize=(25, 10))

sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.barplot(x="id_31", y="id_31Count", data=group_top)

xt = plt.xticks(rotation=60)
# This variable is NOT listed as categorical, but clearly is

plt.figure(figsize=(10, 5))

sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.countplot(x='ProductCD', data=train)
# Plot XII: TransactionDT, TransactionAmt

f, axes = plt.subplots(2, 1, figsize=(15, 10))



dt = sns.distplot(train.TransactionDT,kde=False, ax=axes[0])

am = sns.distplot(train.TransactionAmt,kde=False, hist_kws={'log':True}, ax=axes[1])
C7_loc = train.columns.get_loc("C7")

C14_loc = train.columns.get_loc("C14")

df_c = train.iloc[:,C7_loc:C14_loc+1] #subset dataframe



cols = df_c.columns



rows = 8

f, axes = plt.subplots(rows, 1, figsize=(15, 20))

for i in range(rows):

    dp = sns.distplot(df_c[cols[i]],kde=False, hist_kws={'log':True}, ax=axes[i])

plt.tight_layout()
D1_loc = train.columns.get_loc("D1")

D15_loc = train.columns.get_loc("D15")

df_d = train.iloc[:,D1_loc:D15_loc+1] #subset dataframe



cols = df_d.columns

rows = 15

f, axes = plt.subplots(rows, 1, figsize=(15, 25))

for i in range(rows):

    #d = sns.distplot(df_d[cols[i]].dropna(),kde=False, hist_kws={'log':True}, ax=axes[i])

    d = sns.distplot(df_d[cols[i]].dropna(), ax=axes[i])

plt.tight_layout()
V1_loc = train.columns.get_loc("V1")

V339_loc = train.columns.get_loc("V339")

df = train.iloc[:,V1_loc:V339_loc+1] #subset dataframe



df.head(20)
id_01_loc = train.columns.get_loc("id_01")

id_11_loc = train.columns.get_loc("id_11")

df = train.iloc[:,id_01_loc:id_11_loc+1] #subset dataframe



cols = df.columns

rows = 11

f, axes = plt.subplots(rows, 1, figsize=(15, 25))

for i in range(rows):

    #d = sns.distplot(df[cols[i]].dropna(),kde=False, ax=axes[i])

    d = sns.distplot(df[cols[i]].dropna(), ax=axes[i])

plt.tight_layout()
id_01_loc = train.columns.get_loc("id_01")

id_11_loc = train.columns.get_loc("id_11")

df = train.iloc[:,id_01_loc:id_11_loc+1] #subset dataframe



cols = df.columns

rows = 11

f, axes = plt.subplots(rows, 1, figsize=(15, 25))

for i in range(rows):

    d = sns.distplot(df[cols[i]].dropna(),kde=False, hist_kws={'log':True}, ax=axes[i])

plt.tight_layout()
f, axes = plt.subplots(1, 3, figsize=(15, 8))

isFraud = sns.countplot(x='isFraud', data=train, ax=axes[0])

ProductCD = sns.countplot(x='ProductCD', hue="isFraud", data=train, ax=axes[1])

DeviceType = sns.countplot(x='DeviceType', hue="isFraud", data=train, ax=axes[2])

plt.tight_layout()
f, axes = plt.subplots(1, 2, figsize=(15, 8))



props = train.groupby("ProductCD")['isFraud'].value_counts(normalize=True).unstack()

p = props.plot(kind='bar', stacked='True', ax=axes[0])



props = train.groupby("DeviceType")['isFraud'].value_counts(normalize=True).unstack()

p = props.plot(kind='bar', stacked='True', ax=axes[1])
# Subset dataframe

fraud = pd.DataFrame()

is_fraud = train[train['isFraud']==1]

fraud['DeviceCount'] = is_fraud.groupby(['DeviceInfo'])['DeviceInfo'].count()

fraud['DeviceInfo'] = fraud.index



# There are too many Devices, so we will subset the top 20

group_top = fraud.sort_values(by='DeviceCount',ascending=False).head(20)



plt.figure(figsize=(25, 10))

sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.barplot(x="DeviceInfo", y="DeviceCount", data=group_top)



font_size= {'size': 'x-large'}

ax.set_title("Fraud transactions by OS", **font_size)

xt = plt.xticks(rotation=60)
# These cards are encoded as float64, and since there are too many values

# we will plot this as distributions (x axis is just a class identifier)



is_fraud = train[train['isFraud']==1]

no_fraud = train[train['isFraud']==0]



f, axes = plt.subplots(4, 1, figsize=(25, 30))



d1 = sns.distplot(no_fraud.card1, color="fuchsia", label="No fraud", ax=axes[0])

l1 = d1.legend()

c1 = sns.distplot(is_fraud.card1, color="black", label = "Fraud", ax=axes[0])

l2 = c1.legend()



d2 = sns.distplot(no_fraud.card2.dropna(), color="fuchsia", label="No fraud", ax=axes[1])

l3 = d2.legend()

c2 = sns.distplot(is_fraud.card2.dropna(), color="black",  label = "Fraud", ax=axes[1])

l4 = c2.legend()



d3 = sns.distplot(no_fraud.card3.dropna(), color="fuchsia", label="No fraud", ax=axes[2])

l5 = d3.legend()

c3 = sns.distplot(is_fraud.card3.dropna(), color="black",  label = "Fraud", ax=axes[2])

l6 = c3.legend()



d5 = sns.distplot(no_fraud.card5.dropna(), color="fuchsia", label="No fraud", ax=axes[3])

l7 = d5.legend()

c5 = sns.distplot(is_fraud.card5.dropna(), color="black", label = "Fraud", ax=axes[3])

l8 = c5.legend()
f, axes = plt.subplots(1, 2, figsize=(18, 10))

sns.set(color_codes=True)

card4 = sns.countplot(x='card4', hue="isFraud", data=train, ax=axes[0])

card6 = sns.countplot(x='card6', hue="isFraud", data=train, ax=axes[1])
f, axes = plt.subplots(1, 2, figsize=(18, 10))



props = train.groupby("card4")['isFraud'].value_counts(normalize=True).unstack()

p = props.plot(kind='bar', stacked='True', ax=axes[0])



props = train.groupby("card6")['isFraud'].value_counts(normalize=True).unstack()

p = props.plot(kind='bar', stacked='True', ax=axes[1])
# Subset fraud dataset

addr = 'addr1'

addrC = 'addr1Count'

fraud = pd.DataFrame()

is_fraud = train[train['isFraud']==1]

fraud[addrC] = is_fraud.groupby([addr])[addr].count()

fraud[addr] = fraud.index



# Subset NOT fraud dataset

NOfraud = pd.DataFrame()

no_fraud = train[train['isFraud']==0]

NOfraud[addrC] = no_fraud.groupby([addr])[addr].count()

NOfraud[addr] = NOfraud.index



# There are too many addr, so we will subset the top 20

group_top_f = fraud.sort_values(by=addrC,ascending=False).head(20)

order_f = group_top_f.sort_values(by=addrC,ascending=False)[addr]



group_top_l = NOfraud.sort_values(by=addrC,ascending=False).head(20)

order_l = group_top_l.sort_values(by=addrC,ascending=False)[addr]



f, axes = plt.subplots(4, 1, figsize=(18, 20))



sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.barplot(x=addr, y=addrC, data=group_top_f, order = order_f, ax=axes[0])

bx = sns.barplot(x=addr, y=addrC, data=group_top_l, order = order_l, ax=axes[1])



az = sns.barplot(x=addr, y=addrC, data=group_top_f, ax=axes[2])

bz = sns.barplot(x=addr, y=addrC, data=group_top_l, ax=axes[3])



font_size= {'size': 'x-large'}

ax.set_title("Fraud transactions by addr1 (ranked)", **font_size)

bx.set_title("Legit transactions by addr1 (ranked)", **font_size)



az.set_title("Fraud transactions by addr1", **font_size)

bz.set_title("Legit transactions by addr1", **font_size)



xt = plt.xticks(rotation=60)

plt.tight_layout()
# Subset fraud dataset

addr = 'addr2'

addrC = 'addr2Count'

fraud = pd.DataFrame()

is_fraud = train[train['isFraud']==1]

fraud[addrC] = is_fraud.groupby([addr])[addr].count()

fraud[addr] = fraud.index



# Subset NOT fraud dataset

NOfraud = pd.DataFrame()

no_fraud = train[train['isFraud']==0]

NOfraud[addrC] = no_fraud.groupby([addr])[addr].count()

NOfraud[addr] = NOfraud.index



# There are too many addr, so we will subset the top 20

group_top_f = fraud.sort_values(by=addrC,ascending=False).head(20)

order_f = group_top_f.sort_values(by=addrC,ascending=False)[addr]



group_top_l = NOfraud.sort_values(by=addrC,ascending=False).head(20)

order_l = group_top_l.sort_values(by=addrC,ascending=False)[addr]



f, axes = plt.subplots(4, 1, figsize=(18, 20))



sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.barplot(x=addr, y=addrC, data=group_top_f, order = order_f, ax=axes[0])

bx = sns.barplot(x=addr, y=addrC, data=group_top_l, order = order_l, ax=axes[1])



az = sns.barplot(x=addr, y=addrC, data=group_top_f, ax=axes[2])

bz = sns.barplot(x=addr, y=addrC, data=group_top_l, ax=axes[3])



font_size= {'size': 'x-large'}

ax.set_title("Fraud transactions by addr1 (ranked)", **font_size)

bx.set_title("Legit transactions by addr1 (ranked)", **font_size)



az.set_title("Fraud transactions by addr1", **font_size)

bz.set_title("Legit transactions by addr1", **font_size)



xt = plt.xticks(rotation=60)

plt.tight_layout()
# Get top 10 

order_p=train.P_emaildomain.value_counts().iloc[:10].index

order_r=train.R_emaildomain.value_counts().iloc[:10].index



f, axes = plt.subplots(1, 2, figsize=(16, 8))



sns.set(color_codes=True)

p_email = sns.countplot(y='P_emaildomain',  hue="isFraud", data=train, order = order_p, ax=axes[0])

r_email = sns.countplot(y='R_emaildomain',  hue="isFraud", data=train, order = order_r, ax=axes[1])

plt.tight_layout()
f, axes = plt.subplots(2, 1, figsize=(12, 20))



props = train.groupby("P_emaildomain")['isFraud'].value_counts(normalize=True).unstack()

p = props.plot(kind='barh', stacked='True', ax=axes[0])



props = train.groupby("R_emaildomain")['isFraud'].value_counts(normalize=True).unstack()

p = props.plot(kind='barh', stacked='True', ax=axes[1])



plt.tight_layout()
M1_loc = train.columns.get_loc("M1")

M9_loc = train.columns.get_loc("M9")

df_m = train.iloc[:,M1_loc:M9_loc+1] #subset dataframe M1-M9

df_m['isFraud'] = train.isFraud 



cols = df_m.columns

f, axes = plt.subplots(3, 3, figsize=(16, 12))

count = 0

for i in range(3): # rows loop

    for j in range(3): # cols loop

        mplot = sns.countplot(x=cols[count], hue = 'isFraud', data=df_m, ax=axes[i,j])

        count += 1 # to loop over col-names

plt.tight_layout()
ms = df_m.columns.tolist()

ms.pop()

rows = len(ms)

f, axes = plt.subplots(rows, 1, figsize=(12, 20))

for i,m in enumerate(ms): 

    props = train.groupby(m)['isFraud'].value_counts(normalize=True).unstack()

    p = props.plot(kind='barh', stacked='True', ax=axes[i])

plt.tight_layout()
# Subset fraud dataset

addr = 'id_30'

addrC = 'id_30Count'

fraud = pd.DataFrame()

is_fraud = train[train['isFraud']==1]

fraud[addrC] = is_fraud.groupby([addr])[addr].count()

fraud[addr] = fraud.index



# Subset NOT fraud dataset

NOfraud = pd.DataFrame()

no_fraud = train[train['isFraud']==0]

NOfraud[addrC] = no_fraud.groupby([addr])[addr].count()

NOfraud[addr] = NOfraud.index



# There are too many OS, so we will subset the top 20

group_top_f = fraud.sort_values(by=addrC,ascending=False).head(20)

order_f = group_top_f.sort_values(by=addrC,ascending=False)[addr]



group_top_l = NOfraud.sort_values(by=addrC,ascending=False).head(20)

order_l = group_top_l.sort_values(by=addrC,ascending=False)[addr]



f, axes = plt.subplots(2, 1, figsize=(18, 20))



sns.set(color_codes=True)

sns.set(font_scale = 1.3)



ax = sns.barplot(y=addr, x=addrC, data=group_top_f, order = order_f, ax=axes[0])

bx = sns.barplot(y=addr, x=addrC, data=group_top_l, order = order_l, ax=axes[1])



font_size= {'size': 'x-large'}

ax.set_title("Fraud transactions by OS (ranked)", **font_size)

bx.set_title("Legit transactions by OS (ranked)", **font_size)



plt.tight_layout()
f, axes = plt.subplots(2, 1, figsize=(12, 20))



props = train.groupby("id_30")['isFraud'].value_counts(normalize=True).unstack()

props = props.sort_values(by=1, ascending = False).head(20) # sort by fraud and get top 20

p = props.plot(kind='barh', stacked='True', ax=axes[0])



props = train.groupby("id_31")['isFraud'].value_counts(normalize=True).unstack()

props = props.sort_values(by=1, ascending = False).head(20) # sort by fraud and get top 20

p = props.plot(kind='barh', stacked='True', ax=axes[1])



plt.tight_layout()
# Let's get the frequency of the cases with higher proportion of Fraud

props_30 = train.groupby("id_30")['isFraud'].value_counts(normalize=True).unstack()

props_30 = props_30.sort_values(by=1, ascending = False).head(20) # sort by fraud and get top 20

id_30_top = props_30.index.tolist()

props_30_c = train.groupby("id_30")['isFraud'].value_counts()

props_30_c.loc[id_30_top]
props_31 = train.groupby("id_31")['isFraud'].value_counts(normalize=True).unstack()

props_31 = props_31.sort_values(by=1, ascending = False).head(20) # sort by fraud and get top 20

id_31_top = props_31.index.tolist()

props_31_c = train.groupby("id_31")['isFraud'].value_counts()

props_31_c.loc[id_31_top]
# This variable is NOT listed as categorical, but clearly is

plt.figure(figsize=(10, 5))

sns.set(color_codes=True)

sns.set(font_scale = 1.3)

ax = sns.countplot(x='ProductCD', hue ="isFraud", data=train)
props = train.groupby("ProductCD")['isFraud'].value_counts(normalize=True).unstack()

p = props.plot(kind='barh', stacked='True')
is_fraud = train[train['isFraud']==1]

no_fraud = train[train['isFraud']==0]



f, axes = plt.subplots(2, 1, figsize=(15, 10))



d1 = sns.distplot(no_fraud.TransactionDT, color="fuchsia", label="No fraud", ax=axes[0])

l1 = d1.legend()

d2 = sns.distplot(is_fraud.TransactionDT, color="black", label = "Fraud", ax=axes[0])

l2 = d1.legend()



t1 = sns.distplot(no_fraud.TransactionAmt.apply(np.log2), color="fuchsia", label="No fraud", ax=axes[1])

l3 = t1.legend()

t2 = sns.distplot(is_fraud.TransactionAmt.apply(np.log2), color="black", label = "Fraud", ax=axes[1])

l4 = t2.legend()



plt.tight_layout()
C7_loc = train.columns.get_loc("C7")

C14_loc = train.columns.get_loc("C14")

df_c = train.iloc[:,C7_loc:C14_loc+1] #subset dataframe

cols = df_c.columns



# run this to allow np.log to work, i.e., prevent zero division

df_c.replace(0, 0.000000001, inplace = True) 



df_c['isFraud'] = train.isFraud 



is_fraud = df_c[train['isFraud']==1]

no_fraud = df_c[train['isFraud']==0]



rows = 8

f, axes = plt.subplots(rows, 1, figsize=(15, 30))



for i in range(rows):

    dp = sns.distplot(no_fraud[cols[i]].apply(np.log), color="fuchsia", ax=axes[i])

    dp = sns.distplot(is_fraud[cols[i]].apply(np.log), color="black", ax=axes[i])

plt.tight_layout()
D1_loc = train.columns.get_loc("D1")

D15_loc = train.columns.get_loc("D15")

df_d = train.iloc[:,D1_loc:D15_loc+1] #subset dataframe

cols = df_d.columns



# run this to allow np.log to work, i.e., prevent zero division

df_d.replace(0, 0.000000001, inplace = True) 



df_d['isFraud'] = train.isFraud 



# log transfrom for visualization

is_fraud = df_d[train['isFraud']==1].apply(np.log)

no_fraud = df_d[train['isFraud']==0].apply(np.log)



rows = 15

f, axes = plt.subplots(rows, 1, figsize=(15, 30))

for i in range(rows):

    dp = sns.distplot(no_fraud[cols[i]].dropna(), color="fuchsia", ax=axes[i])

    dp = sns.distplot(is_fraud[cols[i]].dropna(), color="black", ax=axes[i])

plt.tight_layout()
id_01_loc = train.columns.get_loc("id_01")

id_11_loc = train.columns.get_loc("id_11")

df = train.iloc[:,id_01_loc:id_11_loc+1] #subset dataframe

cols = df.columns



# run this to allow np.log to work, i.e., prevent zero division

df.replace(0, 0.000000001, inplace = True) 



df['isFraud'] = train.isFraud 



# log transfrom for visualization

is_fraud = df[train['isFraud']==1].apply(np.log)

no_fraud = df[train['isFraud']==0].apply(np.log)



# run this to avoid runtime error (log is undefined for inf/NaN values in 'isFraud')

is_fraud.drop(columns=['isFraud'], inplace=True)

no_fraud.drop(columns=['isFraud'], inplace=True)



rows = 11

f, axes = plt.subplots(rows, 1, figsize=(15, 25))

for i in range(rows):

    dp = sns.distplot(no_fraud[cols[i]].dropna(), color="fuchsia", ax=axes[i])

    dp = sns.distplot(is_fraud[cols[i]].dropna(), color="black", ax=axes[i])

plt.tight_layout()
id_01_loc = train.columns.get_loc("id_01")

id_11_loc = train.columns.get_loc("id_11")

df = train.iloc[:,id_01_loc:id_11_loc+1] #subset dataframe

cols = df.columns



# run this to allow np.log to work, i.e., prevent zero division

df.replace(0, 0.000000001, inplace = True) 



df['isFraud'] = train.isFraud 



# log transfrom for visualization

is_fraud = df[train['isFraud']==1].apply(np.log)

no_fraud = df[train['isFraud']==0].apply(np.log)

# Here I subset the dataset by the % difference between Fraud and Not-Fraud transactions

from sklearn import preprocessing



#subset dataframe

V1_loc = train.columns.get_loc("V1")

V339_loc = train.columns.get_loc("V339")

df = train.iloc[:,V1_loc:V339_loc+1] 

cols = df.columns



#scale values

scaler = preprocessing.MinMaxScaler()

scaled_array = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=df.columns)

scaled_df['isFraud'] = train.isFraud 



# compute percentage difference between Fraud/Not-fraud transactions

group_means=scaled_df.groupby('isFraud').mean()

group_means_t = group_means.transpose()

group_means_t['delta_percentage'] = ((group_means_t.iloc[:,1] - group_means_t.iloc[:,0]) / ((group_means_t.iloc[:,1] + group_means_t.iloc[:,0]) / 2)) * 100
# Let's limit the plots to the cases where Fraud differs by 100% to Not-Fraud

# i.e., values that double 

plus_100 = group_means_t[group_means_t["delta_percentage"] >= 100]

plus_100_index = plus_100.index.tolist()

len(plus_100)
# This will plot and format 52! barplots, so it may take while tu run (few minutes)

df['isFraud'] = train.isFraud 

cols = plus_100_index

rows = 13

columns = 4

f, axes = plt.subplots(rows, columns, figsize=(20, 35))

count = 0

for i in range(rows): # rows loop

    for j in range(columns): # cols loop

        mplot = sns.barplot(x="isFraud", y=cols[count], data=df, ax=axes[i,j])

        count += 1 # to loop over col-names

plt.tight_layout()
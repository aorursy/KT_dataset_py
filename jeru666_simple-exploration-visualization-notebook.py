import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import missingno as msno



from subprocess import check_output



df_train = pd.read_csv('../input/train.csv', parse_dates=['date'], dtype={'id':np.uint32, 'store_nbr':np.uint8, 'item_nbr': np.uint32, 'onpromotion': np.bool, 'unit_sales': np.float32})



#df_test = pd.read_csv('../input/test.csv', parse_dates=['date'], dtype={'id':np.uint32, 'store_nbr':np.uint8, 'item_nbr': np.uint32, 'onpromotion': np.bool})



#df_sample = pd.read_csv('../input/sample_submission.csv')

df_stores = pd.read_csv('../input/stores.csv')

df_items = pd.read_csv('../input/items.csv')

df_transactions = pd.read_csv('../input/transactions.csv')

df_oil = pd.read_csv('../input/oil.csv')

df_holidays_events = pd.read_csv('../input/holidays_events.csv')
#df_test.head()

#df_test.isnull().values.any()



#--- for integer type columns ---

def change_datatype(df):

    float_cols = list(df.select_dtypes(include=['int']).columns)

    for col in float_cols:

        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):

            df[col] = df[col].astype(np.int8)

        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):

            df[col] = df[col].astype(np.int16)

        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):

            df[col] = df[col].astype(np.int32)

        else:

            df[col] = df[col].astype(np.int64)



#--- for float type columns ---

def change_datatype_float(df):

    float_cols = list(df.select_dtypes(include=['float']).columns)

    for col in float_cols:

        df[col] = df[col].astype(np.float32)            
print(df_stores.shape)

df_stores.head()
df_stores.isnull().values.any()
pp = pd.value_counts(df_stores.dtypes)

fig = plt.figure(figsize=(10,4))

pp.plot.bar(color='green')

plt.show()



print(df_stores.dtypes.unique())

print(df_stores.dtypes.nunique())
df_stores.store_nbr.nunique()
#--- Various cities distribution ---

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

ax = sns.countplot(y=df_stores['city'], data=df_stores) 
#--- Various states distribution ---

fig, ax = plt.subplots()

fig.set_size_inches(8, 8)

ax = sns.countplot(y=df_stores['state'], data=df_stores) 
#--- Various types ---

fig, ax = plt.subplots()

fig.set_size_inches(10, 7)

ax = sns.countplot(x="type", data=df_stores, palette="Set3")
ct = pd.crosstab(df_stores.city, df_stores.type)



ct.plot.bar(figsize = (12, 6), stacked=True)

plt.legend(title='type')



plt.show()
ct = pd.crosstab(df_stores.state, df_stores.type)

ct.plot.bar(figsize = (12, 6), stacked=True)

plt.legend(title='type')

plt.show()
df_stores.cluster.sum()
fig, ax = plt.subplots()

fig.set_size_inches(12, 7)

ax = sns.countplot(x="cluster", data=df_stores)
mm = (df_stores.groupby(['city']).sum())



fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x = mm.index, y= "cluster", data = mm)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)
ct = pd.crosstab(df_stores.city, df_stores.cluster)

ct.plot.bar(figsize = (12, 6), stacked=True)

plt.legend(title='cluster')

plt.show()
ct = pd.crosstab(df_stores.state, df_stores.cluster)

ct.plot.bar(figsize = (12, 6), stacked=True)

plt.legend(title='cluster')

plt.show()
obj_cols = list(df_stores.select_dtypes(include=['object']).columns)

for col in obj_cols:

    df_stores[col], _ = pd.factorize(df_stores[col])

    

df_stores.head(10)
mem = df_stores.memory_usage(index=True).sum()

print("Memory consumed by stores dataframe initially  :   {} MB" .format(mem/ 1024**2))



change_datatype(df_stores)

change_datatype_float(df_stores)



mem = df_stores.memory_usage(index=True).sum()

print("\n Memory consumed by stores dataframe later  :   {} MB" .format(mem/ 1024**2))
print(df_items.shape)

df_items.head()
df_items.isnull().values.any()
fig = plt.figure(figsize=(10,4))

pp = pd.value_counts(df_items.dtypes)

pp.plot.bar(color='red')

plt.show()



print(df_items.dtypes.unique())

print(df_items.dtypes.nunique())
df_items.item_nbr.nunique()
fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.countplot(y = "family", data = df_items)
ct = pd.crosstab(df_items.family, df_items.perishable)

ct.plot.bar(figsize = (12, 7), stacked=True)

plt.legend(title='perishable')

plt.show()
''' 

mc = (df_items.groupby(['family']).sum())

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x = mc.index, y= "perishable", data = mc)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 9)

'''
xc = df_items.groupby(['family'])['class'].nunique()

fig, ax = plt.subplots()

fig.set_size_inches(12, 6)

xc.plot.bar(color='magenta')

plt.show()
obj_cols = list(df_items.select_dtypes(include=['object']).columns)

for col in obj_cols:

    df_items[col], _ = pd.factorize(df_items[col])

    

df_items.head(10)
mem = df_items.memory_usage(index=True).sum()

print("Memory consumed by items dataframe initially  :   {} MB" .format(mem/ 1024**2))



change_datatype(df_items)

change_datatype_float(df_items)



mem = df_items.memory_usage(index=True).sum()

print("\n Memory consumed by items dataframe later  :   {} MB" .format(mem/ 1024**2))
print(df_transactions.shape)

df_transactions.head()
df_transactions.isnull().values.any()
fig = plt.figure(figsize=(10,4))

pp = pd.value_counts(df_transactions.dtypes)

pp.plot.bar(color='orange')

plt.show()



print(df_transactions.dtypes.unique())

print(df_transactions.dtypes.nunique())
pc = (df_transactions.groupby(['store_nbr']).sum())

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x = pc.index, y= "transactions", data = pc)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 9)
g = sns.factorplot(x="store_nbr", y="transactions", size = 12, data=df_transactions)

df_transactions.date = pd.to_datetime(df_transactions.date)

df_transactions.date.dtype
max_transaction = df_transactions['transactions'].max()

min_transaction = df_transactions['transactions'].min()



print(df_transactions.store_nbr[df_transactions['transactions'] == max_transaction])

print(df_transactions.store_nbr[df_transactions['transactions'] == min_transaction])
top_trans = df_transactions.nlargest(100, 'transactions')

print(top_trans.store_nbr.unique())
top_trans = df_transactions.nlargest(500, 'transactions')

print(top_trans.store_nbr.unique())
top_trans = df_transactions.nlargest(1000, 'transactions')

print(top_trans.store_nbr.unique())
mem = df_transactions.memory_usage(index=True).sum()

print("Memory consumed by transactions dataframe initially  :   {} MB" .format(mem/ 1024**2))



change_datatype(df_transactions)

change_datatype_float(df_transactions)



mem = df_transactions.memory_usage(index=True).sum()

print("\nMemory consumed by transactions dataframe later  :   {} MB" .format(mem/ 1024**2))
print(df_holidays_events.shape)

df_holidays_events.head()
df_holidays_events.isnull().values.any()
fig = plt.figure(figsize=(10,4))

pp = pd.value_counts(df_holidays_events.dtypes)

pp.plot.bar(color='violet')

plt.show()



print(df_holidays_events.dtypes.unique())

print(df_holidays_events.dtypes.nunique())
print(df_holidays_events.type.unique())

df_holidays_events.type.value_counts()
fig, ax = plt.subplots()

fig.set_size_inches(8, 6)

ax = sns.countplot( y="type", data=df_holidays_events, palette="RdBu")
print(df_holidays_events.locale.unique())

df_holidays_events.locale.value_counts()
fig, ax = plt.subplots()

fig.set_size_inches(9, 5)

ax = sns.countplot( x="locale", data=df_holidays_events, palette="muted")
ct = pd.crosstab(df_holidays_events.type, df_holidays_events.locale)

ct.plot.bar(figsize = (12, 7), stacked=True)

plt.legend(title='locale')

plt.show()

'''

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

sns.countplot( x="type", hue="locale", data=df_holidays_events, palette="muted")

'''
df_holidays_events.transferred.value_counts()
df_holidays_events.transferred.hist()
fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

sns.countplot( x="type", hue="transferred", data=df_holidays_events)
g = sns.factorplot(x="who", y="survived", col="class",

                    data=df_holidays_events, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

obj_cols = list(df_holidays_events.select_dtypes(include=['object']).columns)

for col in obj_cols:

    df_holidays_events[col], _ = pd.factorize(df_holidays_events[col])

    

df_holidays_events.head(10)
mem = df_holidays_events.memory_usage(index=True).sum()

print("Memory consumed by transactions dataframe initially  :   {} MB" .format(mem/ 1024**2))



change_datatype(df_holidays_events)

change_datatype_float(df_holidays_events)



mem = df_holidays_events.memory_usage(index=True).sum()

print("\nMemory consumed by transactions dataframe later  :   {} MB" .format(mem/ 1024**2))
print(df_oil.shape)

df_oil.head()
df_oil.isnull().values.any()
fig = plt.figure(figsize=(10,4))

pp = pd.value_counts(df_oil.dtypes)

pp.plot.bar(color='pink')

plt.show()



print(df_oil.dtypes.unique())

print(df_oil.dtypes.nunique())
df_oil.date = pd.to_datetime(df_oil.date)

df_oil.date.dtype
print('Maximum price date : ', df_oil.date[df_oil['dcoilwtico'] == df_oil['dcoilwtico'].max()])

print('Minimum price date : ', df_oil.date[df_oil['dcoilwtico'] == df_oil['dcoilwtico'].min()])
ax = sns.boxplot(x=df_oil["dcoilwtico"])
#f, ax = plt.subplots(figsize=(12, 5))

#sns.tsplot(data = df_oil['dcoilwtico'], time = df_oil['date'], err_style="ci_bars", interpolate=False)



f, ax = plt.subplots(figsize=(15, 7))

sns.tsplot(data = df_oil['dcoilwtico'], time = df_oil['date'],ci="sd")

print(df_train.shape)

df_train.head()
mem = df_train.memory_usage(index=True).sum()

print("Memory consumed by train dataframe : {} MB" .format(mem/ 1024**2))
df_train.isnull().values.any()
df_train.columns[df_train.isnull().any()].tolist()
df_train[[ 'onpromotion']] = df_train[['onpromotion']].fillna(np.int(-1))
pp = pd.value_counts(df_train.dtypes)

pp.plot.bar()

plt.show()



print(df_train.dtypes.unique())

print(df_train.dtypes.nunique())
'''

f, ax = plt.subplots(figsize=(15, 7))

sns.tsplot(data = df_train['unit_sales'], time = df_train['date'], ci="sd")

'''
'''

stores_unitsales = (df_train.groupby(['store_nbr']).sum())

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x = stores_unitsales.index, y= "unit_sales", data = stores_unitsales)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)

'''

''' 

items_unitsales = (df_train.groupby(['item_nbr']).sum())

 

fig, ax = plt.subplots()

fig.set_size_inches(12, 8)

ax = sns.barplot(x = items_unitsales.index, y= "unit_sales", data = items_unitsales)

ax.set_xticklabels(ax.get_xticklabels(), rotation = 75, fontsize = 12)

''' 
''' 

print(df_train.onpromotion.nunique())

print(df_train.onpromotion.unique())

'''
#--- non Nan rows ---

#df_train.onpromotion.count()
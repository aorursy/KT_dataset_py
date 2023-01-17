import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

import sys

import datetime

import gc

import matplotlib.pyplot as plt

import seaborn as sns

from textwrap import wrap

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

history = pd.read_csv("../input/historical_transactions.csv",parse_dates=['purchase_date'])

new =pd.read_csv("../input/new_merchant_transactions.csv",parse_dates=['purchase_date'])

history = history.loc[history.authorized_flag =="Y",]
print(history.purchase_date.min(),history.purchase_date.max())

print(new.purchase_date.min(),new.purchase_date.max())
print(new.purchase_date[new.month_lag==1].min(),new.purchase_date[new.month_lag==1].max())

print(new.purchase_date[new.month_lag==2].min(),new.purchase_date[new.month_lag==2].max())
print(history.loc[history.month_lag==0,'purchase_date'].min())

print(history.loc[history.month_lag==0,'purchase_date'].max())
cardreferencedate = history.loc[history.month_lag==0,].groupby('card_id').agg({'purchase_date' : 'max'})

cardreferencedate.reset_index(inplace=True)

cardreferencedate['reference_date'] = cardreferencedate.purchase_date.apply(lambda x :x+ pd.offsets.MonthBegin())

cardreferencedate['reference_date']=cardreferencedate['reference_date'].apply(lambda x: x.replace(hour=0, minute=0, second=0))

cardreferencedate.head()
cardreferencedate.loc[:,'reference_date'].value_counts().sort_index().plot(kind='bar')
new.loc[new.card_id.isin(cardreferencedate.card_id[cardreferencedate.reference_date=='2017-03-01 00:00:00']),]
print(new.loc[new.card_id.isin(cardreferencedate.card_id[cardreferencedate.reference_date=='2017-03-01 00:00:00']),'purchase_date'].min())

print(new.loc[new.card_id.isin(cardreferencedate.card_id[cardreferencedate.reference_date=='2017-03-01 00:00:00']),'purchase_date'].max())
print("Number of cards with reference date", len(cardreferencedate.index))

print("Number of cards in train",len(train.index))

print("Number of cards in test",len(test.index))

print("Number of unique cards in history",len(history.card_id.unique()))
Nozeromonthlag = history.loc[~history.card_id.isin(cardreferencedate.card_id),'card_id'].unique()

len(Nozeromonthlag)
history.loc[history.card_id=='C_ID_21117571cf','month_lag'].value_counts()
history.loc[history.card_id=='C_ID_21117571cf',]
new.loc[new.card_id=='C_ID_21117571cf',]
Nozeromonthlagrefdate = history.loc[~history.card_id.isin(cardreferencedate.card_id),].groupby('card_id').agg({'month_lag' : 'max','purchase_date':'max'})

Nozeromonthlagrefdate.reset_index(inplace=True)

Nozeromonthlagrefdate['month_add'] = Nozeromonthlagrefdate.month_lag.apply(lambda x : abs(x))

Nozeromonthlagrefdate['reference_date'] = Nozeromonthlagrefdate.apply(lambda x: x['purchase_date'] + pd.DateOffset(months = x['month_add']), axis=1)

Nozeromonthlagrefdate['reference_date'] = Nozeromonthlagrefdate.reference_date.apply(lambda x :x+ pd.offsets.MonthBegin())

Nozeromonthlagrefdate['reference_date'] = Nozeromonthlagrefdate['reference_date'].apply(lambda x: x.replace(hour=0, minute=0, second=0))

sum(Nozeromonthlagrefdate.card_id.isin(new.card_id))
Nozeromonthlagrefdate.loc[~Nozeromonthlagrefdate.card_id.isin(new.card_id),'reference_date'].value_counts().plot(kind='bar')
Nozeromonthlag_nonewtransaction_card_id =Nozeromonthlagrefdate.card_id[~Nozeromonthlagrefdate.card_id.isin(new.card_id)]
Nozeromonthlag_nonewtransaction_card_agg=  history.loc[history.card_id.isin(Nozeromonthlag_nonewtransaction_card_id),].groupby(['card_id']).agg({'card_id': 'count','month_lag': ['min','max'],'purchase_date': ['min','max'] })
Nozeromonthlag_nonewtransaction_card_agg.head(50)
print(train.target.mean())

print(train.target[train.card_id.isin(Nozeromonthlag_nonewtransaction_card_id)].mean())
sns.boxplot(y=train.target[train.card_id.isin(Nozeromonthlag_nonewtransaction_card_id)])
zeromonthlagmissinginnew= cardreferencedate.card_id[~cardreferencedate.card_id.isin(new.card_id)]

len(zeromonthlagmissinginnew)
cardreferencedate.drop(columns='purchase_date',inplace=True)

cardreferencedate['category_month_lag'] =np.where(cardreferencedate.card_id.isin(zeromonthlagmissinginnew),1,0)

Nozeromonthlagrefdate['category_month_lag']= np.where(Nozeromonthlagrefdate.card_id.isin(Nozeromonthlag_nonewtransaction_card_id),3,2)

Nozeromonthlagrefdate.drop(columns=['month_lag','purchase_date','month_add'],inplace=True)

cardreferencedate= pd.concat([cardreferencedate,Nozeromonthlagrefdate])

cardreferencedate.to_csv("Cardreferencedate.csv",index=False)

len(cardreferencedate.index)
cardreferencedate = pd.merge(cardreferencedate,train.loc[:,['card_id','target']],on='card_id',how='left')
cardreferencedate.head()
sns.set(rc={'figure.figsize':(24,12)})

p1= sns.boxplot(x=cardreferencedate.reference_date,y=cardreferencedate.target)

labels = [item.get_text() for item in p1.get_xticklabels()]

labels =[ '\n'.join(wrap(l, 10)) for l in labels ]

p1= p1.set_xticklabels(labels, rotation=90)
sns.set(rc={'figure.figsize':(24,12)})

p1= sns.boxplot(x=cardreferencedate.category_month_lag,y=cardreferencedate.target)

# labels = [item.get_text() for item in p1.get_xticklabels()]

# labels =[ '\n'.join(wrap(l, 10)) for l in labels ]

# p1= p1.set_xticklabels(labels, rotation=90)
sns.set(rc={'figure.figsize':(24,6)})  

plt.subplot(1,2,1)

p1=cardreferencedate.loc[cardreferencedate.card_id.isin(train.card_id),'reference_date'].value_counts().sort_index().plot(kind='bar')

p1.set_title("Credit cardsreference date - Train")

plt.subplot(1,2,2)

p2=cardreferencedate.loc[cardreferencedate.card_id.isin(test.card_id),'reference_date'].value_counts().sort_index().plot(kind='bar')

p2.set_title("Credit cardsreference date - Test")
testmissingnew = test.card_id[~test.card_id.isin(new.card_id)]

len(testmissingnew)
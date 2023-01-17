import numpy as np

import pandas as pd

#visualisation 



import matplotlib.pyplot as plt

import time

from datetime import date

from datetime import datetime

import datetime as dt





from wordcloud import WordCloud

from ipywidgets import interact

from collections import defaultdict
#download the files and become in dataframe(pandas)

path = "../input/"



bakery = pd.read_csv(path + "BreadBasket_DMS.csv", sep=",")
print("  - Bakery: \nbakery:", bakery.shape)

print("Head:")

print(bakery.head(),"\n")
print('The number of products that are sold:',bakery['Item'].drop_duplicates().count())
fig, axes=plt.subplots(nrows=2, figsize=(10,8))



#WordCloud Graph

items_dict=bakery.groupby('Item')['Item'].count().sort_values(ascending=False).to_dict()

wordcloud = WordCloud()

wordcloud.generate_from_frequencies(frequencies=items_dict)

axes[0].imshow(wordcloud, interpolation="bilinear")

axes[0].axis("off")



# Frequency Bar

bakery.groupby('Item')['Item'].count().sort_values(ascending=False)[0:19].plot.bar(ax=axes[1])

plt.title('Frequency the 20 most popular items')

plt.show()
bakery['Date'] = pd.to_datetime(bakery['Date'],format='%Y-%m-%d')

bakery['Year'] = bakery['Date'].dt.year

bakery['Month'] = bakery['Date'].dt.month

bakery['Day'] = bakery['Date'].dt.day

bakery['Weekday'] = bakery['Date'].dt.weekday

bakery['Hour'] = pd.to_datetime(bakery['Time'],format='%H:%M:%S').dt.hour
def draw_plots(freq,i):

    aux1_df=pd.DataFrame(bakery.groupby(freq)['Transaction'].count())

    #plot 1

    aux1_df.plot.line(ax=axes[i,0])

    axes[i,0].set_title('Frequency of Transactions per %s' %freq)

    #plot 2

    bakery_10_popItems.groupby([freq,'Item'])['Transaction'].count().unstack().fillna(0).plot.bar(stacked=True, ax=axes[i,1])

    if freq=='Date':

        x=list(np.arange(1,len(aux1_df)+1,20))

        aux1_df['Range']=range(1,len(aux1_df)+1)

        a=aux1_df[aux1_df['Range'].isin(x)]

        labels=(a.index)

        labels=labels.format(str)[1:]

        plt.sca(axes[i, 1])

        plt.xticks(x, labels, rotation=30)

    axes[i,1].set_title('Stacked bar per %s' %freq)
popular_items_10=bakery.groupby('Item')['Transaction'].count().sort_values(ascending=False)[0:10]

popular_items_10=pd.DataFrame(popular_items_10)

bakery_10_popItems=bakery[bakery['Item'].isin(popular_items_10.index)]

# features from plot

fig, axes=plt.subplots(nrows=6, ncols=2,figsize=(20,20))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.50)



freq=['Date','Year', 'Month', 'Day','Weekday','Hour']





for (f,i) in zip(freq, np.arange(6)):

    draw_plots(f,i)



Nothing=bakery[bakery['Item']=='NONE'].groupby('Date')['Transaction'].count()

print('Mean:',Nothing.mean())

print('Std:', Nothing.std())

fig, ax=plt.subplots(nrows=1, figsize=(10,5))

Nothing.plot.line(ax=ax)

plt.show()
bakery['Date'].max()-bakery['Date'].min()
items_dict.keys()
bakery['n']=bakery['Item'].replace(items_dict)
fig, ax=plt.subplots(nrows=1, figsize=(10,5))

baker_less_freq=bakery[bakery['n']<23].groupby(['Date','n'])['n'].count().unstack().fillna(0)

bakery[bakery['n']<23].groupby(['Date','n'])['n'].count().unstack().fillna(0).plot.line(ax=ax)

ax.legend(loc='right')

plt.title('Elements with less frequency ')

plt.show()
def number_items_transaction(n,i,j):

    transaction_23_list=bakery[bakery['n']==n].Transaction

    print('number of elements with frequency %d: %d' %(n,len(transaction_23_list)) )

    if len(transaction_23_list)==0:

        plt.sca(axes[i, j])

        axes[i,j].set_title('#Items per transaction with items bought {}'.format(n))

        plt.text(0.4, 0.5, "No Values", size=10, rotation=20.,

         ha="center", va="center",

         bbox=dict(boxstyle="round",

                   ec=(1., 0.5, 0.5),

                   fc=(1., 0.8, 0.8),

                   ))

    else:

        bakery_23=bakery[bakery['Transaction'].isin(transaction_23_list)]

        aux=pd.DataFrame(bakery_23.groupby('Transaction')['Item'].count()).groupby('Item')['Item'].count()

        #print('number of elements sold: %d'%len(bakery_23))

        pd.DataFrame(aux).plot.bar(ax=axes[i,j])

        axes[i,j].set_title('#Items per transaction with items bought {}'.format(n))

        #plt.title('Transaction with #elements per transaction')

        axes[i,j].get_legend().remove()

  
fig, axes=plt.subplots(nrows=7, ncols=3, figsize=(15,10))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1.3)



n_list=list(np.arange(1,22))

i_list = sorted(list(np.arange(7))*3)

j_list = list(np.arange(3))*7



for (n,i,j) in zip(n_list,i_list,j_list):

    number_items_transaction(n,i,j)
a=bakery_10_popItems.groupby(['Date','Item'])['Transaction'].count().unstack().fillna(0)
a.corr(method='pearson')
def transact_condition(items,i):

    trans_list=list(bakery[bakery['Item']==items].Transaction)

    aux1_df=bakery[bakery['Transaction'].isin(trans_list)]

    aux2_df=aux1_df

    aux1_df=pd.DataFrame(aux1_df.groupby('Transaction')['Item'].count()).groupby('Item')['Item'].count()

    aux1_df=pd.DataFrame(aux1_df)

    aux1_df['Prob']=aux1_df['Item']/sum(aux1_df['Item'])

    aux2_df=pd.DataFrame(aux2_df.groupby('Item')['Item'].count().sort_values(ascending=False))

    aux2_df=pd.DataFrame(aux2_df)

    aux2_df['Prob']=aux2_df['Item']/sum(aux2_df['Item'])

    

    #plot 1

    pd.DataFrame(aux1_df['Prob']).plot.bar(ax=axes[i,0])

    axes[i,0].set_title('P(#items per transaction|%s)' %items)

    axes[i,0].set_xlabel(items)

    axes[i,0].get_legend().remove()

    

    

    #2.

    aux2_df['Prob'][1:20].plot.bar(ax=axes[i,1])

    #x=np.arange(0,20)

    labels=aux2_df[1:20].index

    plt.sca(axes[i, 1])

    #axes[i,1].set_xticklabels(labels,rotation=40)

    plt.xticks(np.arange(20),labels,rotation=60)

    axes[i,1].set_title('P(item belong to transaction|%s)' %items)

    axes[i,1].set_xlabel(' ')

    #plt.show()
# features from plot

fig, axes=plt.subplots(nrows=9, ncols=2,figsize=(25,40))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.8)

items=['Coffee','Bread','Tea','Cake','Pastry','Sandwich','NONE','Medialuna','Hot chocolate']



for (it,i) in zip(items, np.arange(9)):

    transact_condition(it,i)

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import pandas as pd

#import numpy as np

import operator

import time

#import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks", color_codes=True)







import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
items=pd.read_csv('../input/item_properties_part1.csv')

items1=pd.read_csv('../input/item_properties_part2.csv')

items=pd.concat([items1,items])



items.head()
items.shape
import datetime

times=[]

for i in items['timestamp']:

    times.append(datetime.datetime.fromtimestamp(i//1000.0))

    
items['timestamp']=times
#items.to_csv('../input/items.csv')

items.head()
events=pd.read_csv('../input/events.csv')
events.head

events.shape
visitors=events["visitorid"].unique()

print('Visitor count on actions:',events["visitorid"].shape)

print('Total unique visitors :',visitors.shape)

#unique visitors are almost half the number of total visitors
events["event"].unique()

print(events["transactionid"].dropna().unique().shape[0])

print(events["transactionid"].dropna().shape[0])

print(events["itemid"].unique().shape)

#events["itemid"].unique()
events_count=events["event"].value_counts()

fig, axs = plt.subplots(ncols=2,figsize=(15, 8))

sns.barplot(events_count.index, events_count.values, ax=axs[0])



events_count=events["event"].value_counts()[1:]

#plt.title('Actions Vs Count')

g=sns.barplot(events_count.index, events_count.values,ax=axs[1])

#g.set_yscale('log')

events_count=events["event"].value_counts()[1:]

plt.title('Add-to-cart V/s Transaction')

sns.barplot(events_count.index, events_count.values)



print(events_count)
data = events.event.value_counts()

labels = data.index

sizes = data.values

explode = (0, 0.05, 0.15)

fig, ax = plt.subplots(figsize=(8,8))



patches, texts, autotexts = ax.pie(sizes, labels=labels, explode=explode, autopct='%1.2f%%', shadow=False, startangle=0) 



ax.axis('equal')

plt.show()
#sns.factorplot("sex", "survival_rate", col="class", data=df, kind="bar")

grouped=events.groupby('event')['itemid'].apply(list)

views=grouped['view']

count_view={}

#for item in set(views[:]):

    #print(item)

#    count_view[item]=views.count(item)

views=np.array(views[:])



unique, counts = np.unique(views, return_counts=True)

count_view=dict(zip(unique, counts))

sorted_count_view =sorted(count_view.items(), key=operator.itemgetter(1),reverse=True)

x=[i[0] for i in sorted_count_view[:5]]

y=[i[1] for i in sorted_count_view[:5]]

sns.barplot(x,y,order=x)

#the most addtocart itemid

addtocart=grouped['addtocart']

count_addtocart={}

# for item in set(addtocart[:]):

#     #print(item)

#     count_addtocart[item]=addtocart.count(item)

addtocart=np.array(addtocart[:])

unique, counts = np.unique(addtocart, return_counts=True)

count_addtocart=dict(zip(unique, counts))



sorted_count_addtocart =sorted(count_addtocart.items(), key=operator.itemgetter(1),reverse=True)

x=[i[0] for i in sorted_count_addtocart[:5]]

y=[i[1] for i in sorted_count_addtocart[:5]]

g=sns.barplot(x,y, order=x)
#the most transaction itemid

transaction=grouped['transaction']

count_transaction={}

# for item in set(transaction[]):

#     #print(item)

#     count_transaction[item]=transaction.count(item)

transaction=np.array(transaction[:])

unique, counts = np.unique(transaction, return_counts=True)

count_transaction=dict(zip(unique, counts))



sorted_count_transaction =sorted(count_transaction.items(), key=operator.itemgetter(1),reverse=True)

x=[i[0] for i in sorted_count_transaction[:5]]

y=[i[1] for i in sorted_count_transaction[:5]]

g=sns.barplot(x,y, order=x)
items = events.itemid.value_counts()

plt.figure(figsize=(16, 9))

plt.hist(items.values, bins=10, log=True,color='red')

plt.xlabel('Number of times item appeared', fontsize=16)

plt.ylabel('Count of displays with item', fontsize=16)

plt.show()

#number of total views, number of avg view by top users(quantile 90% and also all users)

grouped=events.groupby('event')['visitorid'].apply(list)

views=grouped['view']

count_view={}

# for item in set(views[:]):

#     #print(item)

#     count_view[item]=views.count(item)



views=np.array(views[:])



unique, counts = np.unique(views, return_counts=True)

count_view=dict(zip(unique, counts))



sorted_count_view =sorted(count_view.items(), key=operator.itemgetter(1),reverse=True)

x=[i[0] for i in sorted_count_view[:5]]

y=[i[1] for i in sorted_count_view[:5]]

sns.barplot(x,y,order=x)
#number of total transactions, number of avg transactions by top users(quantile 90% and also all users)

transaction=grouped['transaction']

count_transaction={}

# for item in set(transaction:

#     #print(item)

#     count_transaction[item]=transaction.count(item)

transaction=np.array(transaction[:])

unique, counts = np.unique(transaction, return_counts=True)

count_transaction=dict(zip(unique, counts))

sorted_count_transaction =sorted(count_transaction.items(), key=operator.itemgetter(1),reverse=True)

x=[i[0] for i in sorted_count_transaction[:5]]

y=[i[1] for i in sorted_count_transaction[:5]]

g=sns.barplot(x,y, order=x)

#number of total addtocart, number of avg addtocart by top users(quantile 90% and also all users)

addtocart=grouped['addtocart']

count_addtocart={}

# for item in set(addtocart[:]):

#     #print(item)

#     count_addtocart[item]=addtocart.count(item)

addtocart=np.array(addtocart[:])

unique, counts = np.unique(addtocart, return_counts=True)

count_addtocart=dict(zip(unique, counts))

sorted_count_addtocart =sorted(count_addtocart.items(), key=operator.itemgetter(1),reverse=True)

x=[i[0] for i in sorted_count_addtocart[:5]]

y=[i[1] for i in sorted_count_addtocart[:5]]

g=sns.barplot(x,y, order=x)
items = events.visitorid.value_counts()

plt.figure(figsize=(16, 9))

plt.hist(items.values, bins=20, log=True,color='red')

plt.xlabel('Number of times visitor appeared', fontsize=16)

plt.ylabel('Count of displays with visitor', fontsize=16)

plt.show()

#most active user(s)   3 plot of each with view,add,transaction events

allevents=list(events['visitorid'])

count_allevents={}

# for item in set(allevents[:]):

#    # print(item)

#     count_allevents[item]=allevents.count(item)

allevents=np.array(allevents)

unique, counts = np.unique(allevents, return_counts=True)

count_allevents=dict(zip(unique, counts))

sorted_count_allevents =sorted(count_allevents.items(), key=operator.itemgetter(1),reverse=True)

x=[i[0] for i in sorted_count_allevents[:5]]

y=[i[1] for i in sorted_count_allevents[:5]]

g=sns.barplot(x,y, order=x)
#most active item(s)   3 plot of each with view,add,transaction events

allevents=list(events['itemid'])

count_allevents={}

# for item in set(allevents[:]):

#    # print(item)

#     count_allevents[item]=allevents.count(item)

allevents=np.array(allevents)

unique, counts = np.unique(allevents, return_counts=True)

count_allevents=dict(zip(unique, counts))

sorted_count_allevents =sorted(count_allevents.items(), key=operator.itemgetter(1),reverse=True)

x=[i[0] for i in sorted_count_allevents[:5]]

y=[i[1] for i in sorted_count_allevents[:5]]

g=sns.barplot(x,y, order=x)
#Create

##########df= as below



#visitorid event count

#1   view 100

#1   addtocart   50

#1   transa    5

#2   view 100

#2   addtocart   50

#2   transa    5

#3   view 100

#3   addtocart   50

#3   transa    5

print(events.head())

items.head()

#ax = sns.catplot(x=x, y='visitorid',hue="event", data=events.iloc[:,:])



##

#x=[[1,2,3],[1,2,3],[12,1,2]]

#y=[1,2,3]

#sns.barplot(x,x)

##

#top 5-10 in each category plots and all category plots(stacked charts in all categories)



events.itemid.value_counts().describe()
events[events.event == 'view'].itemid.value_counts().describe()
events[events.event == 'addtocart'].itemid.value_counts().describe()
events[events.event == 'transaction'].itemid.value_counts().describe()
corr = events[events.columns].corr()

sns.heatmap(corr,annot = True)

#time vs event
import scipy.sparse as sp

from scipy.sparse import vstack

from scipy import sparse

from scipy.sparse.linalg import spsolve



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from sklearn.model_selection import train_test_split

print(check_output(["ls", "../input"]).decode("utf8"))

events = pd.read_csv('../input/events.csv')

category_tree = pd.read_csv('../input/category_tree.csv')

items1 = pd.read_csv('../input/item_properties_part1.csv')

items2 = pd.read_csv('../input/item_properties_part2.csv')

items = pd.concat([items1, items2])
n_users = events['visitorid'].unique().shape[0]

n_items = items['itemid'].max()

print (str(n_users) +" " +  str(n_items))

user_to_item_matrix = sp.dok_matrix((n_users+1, n_items+2), dtype=np.int8)
action_weights = [1,2,3]

for row in events.itertuples():

#    if row[2] not in user_with_buy:

#        continue

#    mapped_user_key = user_with_buy[row[2]]

    mapped_user_key = row[2]

    if row.event == 'view':

            user_to_item_matrix[mapped_user_key, row[4]] = action_weights[0]

    elif row.event == 'addtocart':

            user_to_item_matrix[mapped_user_key, row[4]] = action_weights[1]        

    elif row.event == 'transaction':

            user_to_item_matrix[mapped_user_key, row[4]] = action_weights[2]

user_to_item_matrix = user_to_item_matrix.tocsr()

print (user_to_item_matrix.shape)
sparsity = float(len(user_to_item_matrix.nonzero()[0]))

sparsity /= (user_to_item_matrix.shape[0] * user_to_item_matrix.shape[1])

sparsity *= 100

print (sparsity)

X_train, X_test = train_test_split(user_to_item_matrix, test_size=0.20)

X_train.shape

X_test.shape
from sklearn.metrics.pairwise import cosine_similarity

# TODO: this is user to user similarity. check item to item similarity as well

cosine_similarity_matrix = cosine_similarity(X_train, X_train, dense_output=False)

cosine_similarity_matrix.setdiag(0)

cosine_similarity_matrix_ll=cosine_similarity_matrix.tolil()

cosine_similarity_matrix.head()
from datetime import datetime, timedelta

from sklearn import preprocessing

from lightfm import LightFM

from scipy.sparse import csr_matrix 

from scipy.sparse import coo_matrix 

from sklearn.metrics import roc_auc_score

import time

from lightfm.evaluation import auc_score

import pickle



def create_data(datapath,start_date,end_date):

    df=pd.read_csv(datapath)

    df=df.assign(date=pd.Series(datetime.fromtimestamp(a/1000).date() for a in df.timestamp))

    df=df.sort_values(by='date').reset_index(drop=True) # for some reasons RetailRocket did NOT sort data by date

    df=df[(df.date>=datetime.strptime(start_date,'%Y-%m-%d').date())&(df.date<=datetime.strptime(end_date,'%Y-%m-%d').date())]

    df=df[['visitorid','itemid','event']]

    return df



def create_implicit_feedback_matrix(df, split_ratio):

    # assume df.columns=['visitorid','itemid','event']

    id_cols=['visitorid','itemid']

    trans_cat=dict()

    for k in id_cols:

        cate_enc=preprocessing.LabelEncoder()

        trans_cat[k]=cate_enc.fit_transform(df[k].values)

    cate_enc=preprocessing.LabelEncoder()

    ratings=cate_enc.fit_transform(df.event) 

    n_users=len(np.unique(trans_cat['visitorid']))

    n_items=len(np.unique(trans_cat['itemid']))    

    split_point=np.int(np.round(df.shape[0]*split_ratio))

    

    rate_matrix=dict()

    rate_matrix['train']=coo_matrix((ratings[0:split_point],(trans_cat['visitorid'][0:split_point],\

                                              trans_cat['itemid'][0:split_point]))\

                             ,shape=(n_users,n_items))

    rate_matrix['test']=coo_matrix((ratings[split_point+1::],(trans_cat['visitorid'][split_point+1::],\

                                              trans_cat['itemid'][split_point+1::]))\

                             ,shape=(n_users,n_items))

    return rate_matrix



def create_implicit_feedback_matrix1(df, split_ratio):

    # assume df.columns=['visitorid','itemid','event']

    split_point=np.int(np.round(df.shape[0]*split_ratio))

    df_train=df.iloc[0:split_point]

    df_test=df.iloc[split_point::]

    df_test=df_test[(df_test['visitorid'].isin(df_train['visitorid']))&\

                     (df_test['itemid'].isin(df_train['itemid']))]

    id_cols=['visitorid','itemid']

    trans_cat_train=dict()

    trans_cat_test=dict()

    for k in id_cols:

        cate_enc=preprocessing.LabelEncoder()

        trans_cat_train[k]=cate_enc.fit_transform(df_train[k].values)

        trans_cat_test[k]=cate_enc.transform(df_test[k].values)

    

    # --- Encode ratings:

    cate_enc=preprocessing.LabelEncoder()

    ratings=dict()

    ratings['train']=cate_enc.fit_transform(df_train.event)

    ratings['test'] =cate_enc.transform(df_test.event)

    

    n_users=len(np.unique(trans_cat_train['visitorid']))

    n_items=len(np.unique(trans_cat_train['itemid']))    

    

    

    rate_matrix=dict()

    rate_matrix['train']=coo_matrix((ratings['train'],(trans_cat_train['visitorid'],\

                                              trans_cat_train['itemid']))\

                             ,shape=(n_users,n_items))

    rate_matrix['test']=coo_matrix((ratings['test'],(trans_cat_test['visitorid'],\

                                              trans_cat_test['itemid']))\

                             ,shape=(n_users,n_items))

    return rate_matrix



if __name__=='__main__':

    start_time = time.time()

    df=create_data('../input/events.csv','2015-5-3','2015-5-18')

    modelLoad=False

    rating_matrix=create_implicit_feedback_matrix1(df,.8)

    if(modelLoad):

        with open('saved_model','rb') as f:

            saved_model=pickle.load(f)

            model=saved_model['model']

    else:

        model=LightFM(no_components=5,loss='warp')

        model.fit(rating_matrix['train'],epochs=100,num_threads=1)

        with open('saved_model','wb') as f:

            saved_model={'model':model}

            pickle.dump(saved_model, f)

    auc_train = auc_score(model, rating_matrix['train']).mean()

    auc_test = auc_score(model, rating_matrix['test']).mean()

    

    #df=df.assign(pred_score=model.predict(df['visitorid'],df['itemid']))

    

    #df_auc=df.groupby(by='visitorid').apply(lambda df: roc_auc_score(df['event'].values,df['pred_score'].values))

    #print('Training auc %0.3f' % numpy.mean([i for i in df_auc.values if i > -1]))
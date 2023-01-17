# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.display.max_columns = 999

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
trainData=pd.read_csv("/kaggle/input/elo-merchant-category-recommendation/train.csv")

newTrans=pd.read_csv("/kaggle/input/elo-merchant-category-recommendation/new_merchant_transactions.csv")

pastTrans=pd.read_csv("/kaggle/input/elo-merchant-category-recommendation/historical_transactions.csv")

merchants=pd.read_csv("/kaggle/input/elo-merchant-category-recommendation/merchants.csv")

newTrans=newTrans.drop(columns=['merchant_category_id','purchase_date'])

pastTrans=pastTrans.drop(columns=['merchant_category_id','purchase_date'])

merchants=merchants[['merchant_id','merchant_category_id','subsector_id','numerical_1','numerical_2','avg_sales_lag12','avg_purchases_lag12','active_months_lag12']]
newTrans.head()
pastTrans.head()
agg_func = {

    'installments': ['mean'],

    'month_lag': ['mean','count'],

    'purchase_amount': ['mean']}

    

newTransSummary = newTrans.groupby(['card_id']).agg(agg_func)

newTransSummary.columns = ['_'.join(col).strip() for col in newTransSummary.columns.values]

newTransSummary.reset_index(inplace=True)

pastTransSummary = pastTrans.groupby(['card_id']).agg(agg_func)

pastTransSummary.columns = ['_'.join(col).strip() for col in pastTransSummary.columns.values]

pastTransSummary.reset_index(inplace=True)
newTrans=newTrans.sort_values(by=['month_lag'], ascending=False)

pastTrans=pastTrans.sort_values(by=['month_lag'], ascending=False)

pastTrans=pastTrans.drop_duplicates('card_id',keep='first')

newTrans=newTrans.drop_duplicates('card_id',keep='first')

newTrans.reset_index(inplace=True)

pastTrans.reset_index(inplace=True)
### Aggregated with the latest and past summary with merchants info

aggregatedNew=newTrans.merge(newTransSummary,on='card_id',how='left')

aggregatedPast=pastTrans.merge(pastTransSummary,on='card_id',how='left')

aggregatedNew=aggregatedNew.merge(merchants,on='merchant_id',how='left')

aggregatedPast=aggregatedPast.merge(merchants,on='merchant_id',how='left')

aggregatedNew=aggregatedNew.drop(columns=['index','merchant_id'])

aggregatedPast=aggregatedPast.drop(columns=['index','merchant_id'])

del pastTrans

del newTrans

del merchants
aggregatedPast.head()
aggregatedPast.columns = [col+'_'+'past' for col in aggregatedPast.columns]

aggregatedAll=aggregatedNew.merge(aggregatedPast,left_on='card_id',right_on='card_id_past',how='outer')

cardId=aggregatedAll['card_id'].tolist()

#aggregatedAll=aggregatedAll.drop(columns=['card_id_past'])
del aggregatedPast

del aggregatedNew
aggregatedAll.head()
aggregatedAll.isna().sum() 

ids=aggregatedAll['card_id_past'].tolist()

aggregatedAll=aggregatedAll.drop(columns=['card_id','card_id_past'])

aggregatedAll['authorized_flag']=aggregatedAll['authorized_flag'].fillna('Unknown')

aggregatedAll['city_id']=aggregatedAll['city_id'].fillna('Unknown')

aggregatedAll['category_1']=aggregatedAll['category_1'].fillna('Unknown')

aggregatedAll['installments']=aggregatedAll['installments'].fillna(0)

aggregatedAll['category_3']=aggregatedAll['category_3'].fillna('Unknown')

aggregatedAll['month_lag']=aggregatedAll['month_lag'].fillna('Unknown')

aggregatedAll['purchase_amount']=aggregatedAll['purchase_amount'].fillna(0.0)

aggregatedAll['category_2']=aggregatedAll['category_2'].fillna('Unknown')

aggregatedAll['state_id']=aggregatedAll['state_id'].fillna('Unknown')

aggregatedAll['subsector_id_x']=aggregatedAll['subsector_id_x'].fillna('Unknown')

aggregatedAll['installments_mean']=aggregatedAll['installments_mean'].fillna(0)

aggregatedAll['month_lag_mean']=aggregatedAll['month_lag_mean'].fillna(0)

aggregatedAll['month_lag_count']=aggregatedAll['month_lag_count'].fillna(0)

aggregatedAll['purchase_amount_mean']=aggregatedAll['purchase_amount_mean'].fillna(0)

aggregatedAll['merchant_category_id']=aggregatedAll['merchant_category_id'].fillna('Unknown')

aggregatedAll['subsector_id_y']=aggregatedAll['subsector_id_y'].fillna('Unknown')

aggregatedAll['numerical_1']=aggregatedAll['numerical_1'].fillna(aggregatedAll['numerical_1'].mean())

aggregatedAll['numerical_2']=aggregatedAll['numerical_2'].fillna(aggregatedAll['numerical_2'].mean())

aggregatedAll['avg_sales_lag12']=aggregatedAll['avg_sales_lag12'].fillna(0.0)

aggregatedAll['avg_purchases_lag12']=aggregatedAll['avg_purchases_lag12'].fillna(0.0)

aggregatedAll['active_months_lag12']=aggregatedAll['active_months_lag12'].fillna(0.0)







aggregatedAll['category_3_past']=aggregatedAll['category_3_past'].fillna('Unknown')

aggregatedAll['category_2_past']=aggregatedAll['category_2_past'].fillna('Unknown')

aggregatedAll['merchant_category_id_past']=aggregatedAll['merchant_category_id_past'].fillna('Unknown')

aggregatedAll['subsector_id_y_past']=aggregatedAll['subsector_id_y_past'].fillna('Unknown')

aggregatedAll['numerical_1_past']=aggregatedAll['numerical_1_past'].fillna(aggregatedAll['numerical_1_past'].mean())

aggregatedAll['numerical_2_past']=aggregatedAll['numerical_2_past'].fillna(aggregatedAll['numerical_2_past'].mean())

aggregatedAll['avg_sales_lag12_past']=aggregatedAll['avg_sales_lag12_past'].fillna(0.0)

aggregatedAll['avg_purchases_lag12_past']=aggregatedAll['avg_purchases_lag12_past'].fillna(0.0)

aggregatedAll['active_months_lag12_past']=aggregatedAll['active_months_lag12_past'].fillna(0.0)

aggregatedAll.isna().sum() 
aggregatedAll.head()
set(aggregatedAll['category_1_past'].tolist())
categoricalColumns=['authorized_flag','category_1','installments','category_3','month_lag','category_2','authorized_flag_past','category_1_past','installments_past','category_3_past','month_lag_past','category_2_past']

toEmbedding=['city_id','state_id','subsector_id_x','merchant_category_id','subsector_id_y',

             'city_id_past','state_id_past','subsector_id_x_past','merchant_category_id_past','subsector_id_y_past']

numericalColumns=['purchase_amount','installments_mean','month_lag_mean','month_lag_count','purchase_amount_mean','numerical_1','numerical_2','avg_sales_lag12','avg_purchases_lag12','active_months_lag12',

                  'purchase_amount_past','installments_mean_past','month_lag_mean_past','month_lag_count_past','purchase_amount_mean_past','numerical_1_past','numerical_2_past','avg_sales_lag12_past','avg_purchases_lag12_past','active_months_lag12_past']
len(categoricalColumns)+len(toEmbedding)+len(numericalColumns)
len(aggregatedAll.columns)
aggregatedAll=aggregatedAll[categoricalColumns+toEmbedding+numericalColumns]
## Latent Dim=5

## On 'city_id','state_id','subsector_id_x','merchant_category_id','subsector_id_y'

allCity=list(set(aggregatedAll['city_id'].tolist()+aggregatedAll['city_id_past'].tolist()))

allState=list(set(aggregatedAll['state_id'].tolist()+aggregatedAll['state_id_past'].tolist()))

allSubsectorx=list(set(aggregatedAll['subsector_id_x'].tolist()+aggregatedAll['subsector_id_x_past'].tolist()))

allMerchantCat=list(set(aggregatedAll['merchant_category_id'].tolist()+aggregatedAll['merchant_category_id_past'].tolist()))

allSubsectory=list(set(aggregatedAll['subsector_id_y'].tolist()+aggregatedAll['subsector_id_y_past'].tolist()))



## Label Encode

cityEncoding={}

cityEncoding2={}

for i in range(len(allCity)):

    cityEncoding[i]=allCity[i]

    cityEncoding2[allCity[i]]=i



stateEncoding={}

stateEncoding2={}



for i in range(len(allState)):

    stateEncoding[i]=allState[i]

    stateEncoding2[allState[i]]=i



subsectorxEncoding={}

subsectorxEncoding2={}



for i in range(len(allSubsectorx)):

    subsectorxEncoding[i]=allSubsectorx[i]

    subsectorxEncoding2[allSubsectorx[i]]=i

    



merchantCatEncoding={}

merchantCatEncoding2={}



for i in range(len(allMerchantCat)):

    merchantCatEncoding[i]=allMerchantCat[i]

    merchantCatEncoding2[allMerchantCat[i]]=i

    



subsectoryEncoding={}

subsectoryEncoding2={}



for i in range(len(allSubsectory)):

    subsectoryEncoding[i]=allSubsectory[i]

    subsectoryEncoding2[allSubsectory[i]]=i



aggregatedAll['ID']=ids

aggregatedAll=aggregatedAll.sample(frac=1)

aggregatedAll.reset_index(drop=True)
aggregatedAll=pd.get_dummies(data=aggregatedAll, columns=categoricalColumns)
aggregatedAll.head()
idLst=aggregatedAll['ID'].tolist()

aggregatedAll=aggregatedAll.drop(columns=['ID'],inplace=False)

aggregatedAll=aggregatedAll.reset_index(drop=True)

trainData2=trainData[['card_id','target']]

aggregatedAll['ID']=idLst

aggregatedAll=aggregatedAll.merge(trainData2,left_on='ID',right_on='card_id',how='right')

aggregatedAll.head()
result=aggregatedAll['target'].tolist()

aggregatedAll=aggregatedAll.drop(columns=['ID','card_id'])

aggregatedAll.head()
aggregatedAll=aggregatedAll.drop(columns=['target'])
aggregatedAll['city_id']=aggregatedAll['city_id'].apply(lambda x:cityEncoding2[x])

aggregatedAll['city_id_past']=aggregatedAll['city_id_past'].apply(lambda x:cityEncoding2[x])



aggregatedAll['state_id']=aggregatedAll['state_id'].apply(lambda x:stateEncoding2[x])

aggregatedAll['state_id_past']=aggregatedAll['state_id_past'].apply(lambda x:stateEncoding2[x])



aggregatedAll['subsector_id_x']=aggregatedAll['subsector_id_x'].apply(lambda x:subsectorxEncoding2[x])

aggregatedAll['subsector_id_x_past']=aggregatedAll['subsector_id_x_past'].apply(lambda x:subsectorxEncoding2[x])



aggregatedAll['merchant_category_id']=aggregatedAll['merchant_category_id'].apply(lambda x:merchantCatEncoding2[x])

aggregatedAll['merchant_category_id_past']=aggregatedAll['merchant_category_id_past'].apply(lambda x:merchantCatEncoding2[x])



aggregatedAll['subsector_id_y']=aggregatedAll['subsector_id_y'].apply(lambda x:subsectoryEncoding2[x])

aggregatedAll['subsector_id_y_past']=aggregatedAll['subsector_id_y_past'].apply(lambda x:subsectoryEncoding2[x])

bigX=np.array(aggregatedAll)
result=np.array([[x] for x in result])
print(bigX.shape)

print(result.shape)
aggregatedAll.head()
print(len(cityEncoding))

print(len(stateEncoding))

print(len(subsectorxEncoding))

print(len(merchantCatEncoding2))

print(len(subsectoryEncoding))

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
x=tf.placeholder(tf.float32,shape=(None, 103))

y=tf.placeholder(tf.float32,shape=(None,1))



cityLab=tf.slice(x, [0,0],[-1, 1])

cityLabPast=tf.slice(x, [0,5],[-1, 1])

cityLab=tf.dtypes.cast(cityLab, tf.int32)

cityLabPast=tf.dtypes.cast(cityLabPast, tf.int32)

cityEmbedding=tf.Variable(tf.random_normal([309,5], stddev=0.1))

embeddedCity=tf.nn.embedding_lookup(cityEmbedding,cityLab)

embeddedCityPast=tf.nn.embedding_lookup(cityEmbedding,cityLabPast)





stateLab=tf.slice(x, [0,2],[-1, 1])

stateLabPast=tf.slice(x, [0,6],[-1, 1])

stateLab=tf.dtypes.cast(stateLab, tf.int32)

stateLabPast=tf.dtypes.cast(stateLabPast, tf.int32)

stateEmbedding=tf.Variable(tf.random_normal([26,5], stddev=0.1))

embeddedState=tf.nn.embedding_lookup(stateEmbedding,cityLab)

embeddedStatePast=tf.nn.embedding_lookup(stateEmbedding,stateLabPast)



subsectorxLab=tf.slice(x, [0,3],[-1, 1])

subsectorxLabPast=tf.slice(x, [0,7],[-1, 1])

subsectorxLab=tf.dtypes.cast(subsectorxLab, tf.int32)

subsectorxLabPast=tf.dtypes.cast(subsectorxLabPast, tf.int32)

subsectorxLabEmbedding=tf.Variable(tf.random_normal([42,5], stddev=0.1))

embeddedSubsectorX=tf.nn.embedding_lookup(subsectorxLabEmbedding,subsectorxLab)

embeddedSubsectorXPast=tf.nn.embedding_lookup(subsectorxLabEmbedding,subsectorxLabPast)



merchantCatLab=tf.slice(x, [0,4],[-1, 1])

merchantCatLabPast=tf.slice(x, [0,8],[-1, 1])

merchantCatLab=tf.dtypes.cast(merchantCatLab, tf.int32)

merchantCatLabPast=tf.dtypes.cast(merchantCatLabPast, tf.int32)

merchantCatLabEmbedding=tf.Variable(tf.random_normal([294,5], stddev=0.1))

embeddedmerchantCat=tf.nn.embedding_lookup(merchantCatLabEmbedding,merchantCatLab)

embeddedmerchantCatPast=tf.nn.embedding_lookup(merchantCatLabEmbedding,merchantCatLabPast)



subsectoryLab=tf.slice(x, [0,5],[-1, 1])

subsectoryLabPast=tf.slice(x, [0,9],[-1, 1])

subsectoryLab=tf.dtypes.cast(subsectoryLab, tf.int32)

subsectoryLabPast=tf.dtypes.cast(subsectoryLabPast, tf.int32)

subsectoryLabEmbedding=tf.Variable(tf.random_normal([41,5], stddev=0.1))

embeddedSubsectorY=tf.nn.embedding_lookup(subsectoryLabEmbedding,subsectoryLab)

embeddedSubsectorYPast=tf.nn.embedding_lookup(subsectoryLabEmbedding,subsectoryLabPast)



additionEncoding=embeddedCity+embeddedCityPast+embeddedState+embeddedStatePast+embeddedSubsectorX+embeddedSubsectorXPast+embeddedmerchantCat+embeddedmerchantCatPast+embeddedSubsectorY+embeddedSubsectorYPast



interCity=tf.multiply(embeddedCity,embeddedCityPast)

interState=tf.multiply(embeddedState,embeddedStatePast)

interSubsectorx=tf.multiply(embeddedSubsectorX,embeddedSubsectorXPast)

interMerchant=tf.multiply(embeddedmerchantCat,embeddedmerchantCatPast)

interSubsectory=tf.multiply(embeddedSubsectorY,embeddedSubsectorYPast)



FMPart=tf.concat(

    [additionEncoding,interCity,interState,interSubsectorx,interMerchant,interSubsectory],2)

FMPart=tf.squeeze(FMPart, [1])
FFfeatures=cityLab=tf.slice(x, [0,10],[-1, 93])

hidden1=tf.layers.dense(FFfeatures, 32,activation=tf.nn.sigmoid)

deepPart=tf.layers.dense(hidden1,16,activation=tf.nn.sigmoid)



DeepFMHidden=tf.concat([FMPart,deepPart],1)

DeepFMHidden2=tf.layers.dense(DeepFMHidden,16,activation=tf.nn.tanh)

DeepFMHidden3=tf.layers.dense(DeepFMHidden2,8,activation=tf.nn.tanh)

DeepFMFinal=tf.layers.dense(DeepFMHidden3,1)
FMPart.shape
deepPart.shape
DeepFMHidden.shape
DeepFMFinal.shape
loss=mse = tf.losses.mean_squared_error(y, DeepFMFinal) 

training=tf.train.AdamOptimizer(0.003).minimize(loss)
## Training
sess=tf.Session()

init=tf.global_variables_initializer()

sess.run(init)



batchSize=128

epochs=50

cnter=0

ttLoss=0.0

for idx in range(epochs):

    cnter=0

    for i in range(int(bigX.shape[0]/batchSize)-1):

        tempX=bigX[i*batchSize:(i+1)*batchSize]

        tempY=result[i*batchSize:(i+1)*batchSize]

        _,temploss=sess.run([training,loss],feed_dict={x:tempX,y:tempY})

        ttLoss+=temploss

        if i%10000==0:

            print('Current Loss: '+ str(ttLoss/10000.0))

            ttLoss=0.0
allPred=[]

for i in range(int(bigX.shape[0]/batchSize)-1):

    tempX=bigX[i*batchSize:(i+1)*batchSize]

    tempY=result[i*batchSize:(i+1)*batchSize]

    _,_,pred=sess.run([training,loss,DeepFMFinal],feed_dict={x:tempX,y:tempY})

    allPred+=list(pred.reshape([128]))

[x[0] for x in result]
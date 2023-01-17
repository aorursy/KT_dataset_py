#import necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#load datasets

df_train=pd.read_csv('../input/demand-forecasting-kernels-only/train.csv',sep=',')

df_test=pd.read_csv('../input/demand-forecasting-kernels-only/test.csv',sep=',')



#drop id column 



if 'id' in df_test.columns:

  df_test.drop(labels='id',axis=1,inplace=True)

  

if 'id' in df_train.columns:

  df_train.drop(labels='id',axis=1,inplace=True)

  



# create sales column for df_test which is to be predicted  

df_test['sales']=np.NaN



df_test['purpose']='test'

df_train['purpose']='train'



#combine df_train and df_test into one dataframe



df=pd.concat([df_train, df_test], axis=0)



df.loc[(df.purpose=='train')&(df.date>'2016-12-31'),'purpose']='validation'





#convert to date_time object



df['date'] = pd.to_datetime(df['date'])



df=df.sort_values(by=['item','store','date'])
# prepare dataframe 

# if we denote sales column with S_t, we add S_(t-1), S_(t-2),.... to this new dataframe



new_df=pd.DataFrame()

for item in df.item.unique():

    for store in df.store.unique():

        #add lagged sales column to  the related part of the original dataframe 

        new=pd.concat([df.loc[(df.store==store)&(df.item==item)]]+[df.loc[(df.store==store)&(df.item==item),'sales'].shift(i) for i in range(1,400)],axis=1)

        new.columns=list(df.columns)+['sales_lagged_{}'.format(i) for i in range(1,400)]

        new_df=pd.concat([new_df,new],axis=0)
#create some new categorical features from date



new_df['weekofyear']=new_df.date.apply(lambda x:x.weekofyear)

new_df['dayofweek']=new_df.date.apply(lambda x:x.dayofweek)

new_df['year']=new_df.date.apply(lambda x:x.year)

new_df['month']=new_df.date.apply(lambda x:x.month)



new_df['weekofyear']=new_df['weekofyear'].astype('category')

new_df['dayofweek']=new_df['dayofweek'].astype('category')

new_df['year']=new_df['year'].astype('category')

new_df['month']=new_df['month'].astype('category')

new_df['store']=new_df.store.astype('category')

new_df['item']=new_df.item.astype('category')
# convert categoricals to dummy variables



new_df_with_dummies=pd.concat([pd.get_dummies(new_df.drop(labels='purpose',axis=1)),new_df.purpose],axis=1)



train=new_df_with_dummies[new_df_with_dummies.purpose=='train']

validation=new_df_with_dummies[new_df_with_dummies.purpose=='validation']

test=new_df_with_dummies[new_df_with_dummies.purpose=='test']



train.dropna(inplace=True)

validation.dropna(inplace=True)



xtrain=train.drop(labels=['sales','date','purpose'],axis=1)

ytrain=train.sales.values

xvalidation=validation.drop(labels=['sales','date','purpose'],axis=1)

yvalidation=validation.sales.values
del new_df, new_df_with_dummies
for col in xtrain.columns:

    if xtrain[col].dtypes=='float':

        xtrain[col]=xtrain[col].astype(np.int16)
for col in xvalidation.columns:

    if xvalidation[col].dtypes=='float':

        xvalidation[col]=xvalidation[col].astype(np.int16)
from sklearn.linear_model import  LassoLarsIC







model_bic = LassoLarsIC(criterion='bic')

model_bic.fit(xtrain,ytrain)
model_aic = LassoLarsIC(criterion='aic')

model_aic.fit(xtrain,ytrain)
from sklearn.linear_model import  LassoLarsCV





model_CV=LassoLarsCV(cv=10,max_iter=1000)

model_CV.fit(xtrain,ytrain)
#best alphas according to the three methods above



print(model_aic.alpha_)

print(model_bic.alpha_)

print(model_CV.alpha_)
# performances on validation set with respect to smape metric



def smape(A, F):

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))



print(smape(model_aic.predict(xvalidation),yvalidation))

print(smape(model_bic.predict(xvalidation),yvalidation))

print(smape(model_CV.predict(xvalidation),yvalidation))
# we now construct a Lasso model with alpha=model_CV.alpha_ on the training set xtrain+xvalidation



xtrain_val=pd.concat([xtrain,xvalidation],axis=0)

ytrain_val=np.concatenate((ytrain,yvalidation),axis=0)



from sklearn.linear_model import Lasso







lasso=Lasso(max_iter=1000, alpha=model_CV.alpha_)

lasso.fit(xtrain_val,ytrain_val)
# predictions with Recursive strategy





predictions={} # two dimensional dictionary #  keys come from  store and item ids

for item in df.item.unique():

    predictions[item]=dict()

    for store in df.store.unique():

        predictions[item][store]=list()

        start=900*(item-1)+90*(store-1)    

        end=900*(item-1)+90*store

        count=0

        for ind1 in range(start,end):

            predictions[item][store].append(lasso.predict(np.concatenate((predictions[item][store][:count][::-1],test.iloc[ind1,count+2:-1].values)).reshape(1,-1))[0])

            count+=1

     
#prepare the submission file





for item in df.item.unique():

    for store in df.store.unique():

        start=900*(item-1)+90*(store-1)    

        end=900*(item-1)+90*store

        df_test.iloc[start:end,3]=predictions[item][store]

        

d={}

d['id']=df_test.index

d['sales']=df_test.sales



pred_submission=pd.DataFrame(d, columns=['id','sales'])



pred_submission.to_csv('submission.csv', index=False)
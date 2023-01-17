import numpy as np

import pandas as pd

#creating a classification model to predict category for records with category other

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.pipeline import FeatureUnion

from sklearn.model_selection import train_test_split

import scipy.sparse as sp

import scipy.stats as stat

from sklearn.linear_model import Ridge, SGDRegressor, LinearRegression, Lasso, ElasticNet

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import GridSearchCV 

from sklearn.feature_selection import VarianceThreshold

from subprocess import check_output

import difflib as dfl

print(check_output(["ls", "../input"]).decode("utf8"))
train=pd.read_table('../input/train.tsv')

test=pd.read_table('../input/test.tsv')
train.category_name.fillna('other/other/other',inplace=True)

train.brand_name.fillna('other',inplace=True)

train.item_description.fillna('no description',inplace=True)

test.category_name.fillna('other/other/other',inplace=True)

test.brand_name.fillna('other',inplace=True)
train['category_name']=train.category_name.apply(lambda x: x.lower())

train['name']=train.name.apply(lambda x: x.lower())

train['brand_name']=train.brand_name.apply(lambda x: x.lower())

train['item_description']=train.item_description.apply(lambda x: x.lower())

test['category_name']=test.category_name.apply(lambda x: x.lower())

test['name']=test.name.apply(lambda x: x.lower())

test['brand_name']=test.brand_name.apply(lambda x: x.lower())

test['item_description']=test.item_description.apply(lambda x: x.lower())
train['main_category']=train.category_name.apply(lambda x: x.split('/')[0])

train['sub_category']=train.category_name.apply(lambda x: x.split('/')[1])

train['item_type']=train.category_name.apply(lambda x: x.split('/')[2])

test['main_category']=test.category_name.apply(lambda x: x.split('/')[0])

test['sub_category']=test.category_name.apply(lambda x: x.split('/')[1])

test['item_type']=test.category_name.apply(lambda x: x.split('/')[2])
%%time

df_category=pd.DataFrame(train.category_name.unique(),columns=['category_name'])

df_category['count']=df_category.category_name.apply(lambda x: len(train.category_name[train.category_name==x]))

df_category['category']=df_category.category_name.apply(lambda x: (x.split('/')[0]+'/'+x.split('/')[1]))



for i,cat in enumerate(df_category.category):

  if df_category.iloc[i,1]<5000:

    df_category.iloc[i,2]=(cat.split('/')[0]+'/'+cat.split('/')[0])

  else:

    df_category.iloc[i,2]=cat

    

df_category['countF']=df_category.category.apply(lambda x: sum(df_category['count'][df_category.category==x]))



for i,cat in enumerate(df_category.category):

  if df_category.iloc[i,3]<5000:

    df_category.iloc[i,2]='other/other'

  else:

    df_category.iloc[i,2]=cat

    

df_category['countF']=df_category.category.apply(lambda x: sum(df_category['count'][df_category.category==x]))

df_category=df_category.drop(['count','countF'],axis=1)



df_train=pd.merge(train,df_category,on='category_name',how='left')

df_test=pd.merge(test,df_category,on='category_name',how='left')

df_test.category.fillna('other/other',inplace=True)

df_train=df_train.drop(['category_name'],axis=1)

df_test=df_test.drop(['category_name'],axis=1)
def get_sparse(df,df1,df2,max_feature=300000):

  

  Cvect=CountVectorizer(binary=True)

  Tvect=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=max_feature)





  vect_name=Tvect.fit(df.name)

  vect_name1=vect_name.transform(df.name)

  vect_name2=vect_name.transform(df1.name)

  vect_name3=vect_name.transform(df2.name)

  #print(vect_name.get_feature_names())

  #print('name features',vect_name1.shape,':',vect_name2.shape)

  

  vect_icond1=sp.csr_matrix(pd.get_dummies(df.item_condition_id,sparse=True).values)

  vect_icond2=sp.csr_matrix(pd.get_dummies(df1.item_condition_id,sparse=True).values)

  vect_icond3=sp.csr_matrix(pd.get_dummies(df2.item_condition_id,sparse=True).values)

  #print('condition features',vect_icond1.shape,':',vect_icond2.shape)

  

  vect_brand=Cvect.fit(df.brand_name)

  vect_brand1=vect_brand.transform(df.brand_name)

  vect_brand2=vect_brand.transform(df1.brand_name)

  vect_brand3=vect_brand.transform(df2.brand_name)

  #print('brand features',vect_brand1.shape,':',vect_brand2.shape)

  

  vect_ship1=sp.csr_matrix(pd.get_dummies(df.shipping,sparse=True).values)

  vect_ship2=sp.csr_matrix(pd.get_dummies(df1.shipping,sparse=True).values)

  vect_ship3=sp.csr_matrix(pd.get_dummies(df2.shipping,sparse=True).values)

  #print('shipping features',vect_ship1.shape,':',vect_ship2.shape)

  

  

  vect_desc=Tvect.fit(df.item_description)

  vect_desc1=vect_desc.transform(df.item_description)

  vect_desc2=vect_desc.transform(df1.item_description)

  vect_desc3=vect_desc.transform(df2.item_description)

  #print('desc features',vect_desc1.shape,':',vect_desc2.shape)

  

  vect_mcat=Cvect.fit(df.main_category)

  vect_mcat1=vect_mcat.transform(df.main_category)

  vect_mcat2=vect_mcat.transform(df1.main_category)

  vect_mcat3=vect_mcat.transform(df2.main_category)

  #print('main category features',vect_mcat1.shape,':',vect_mcat2.shape)

  

  

  vect_scat=Cvect.fit(df.sub_category)

  vect_scat1=vect_scat.transform(df.sub_category)

  vect_scat2=vect_scat.transform(df1.sub_category)

  vect_scat3=vect_scat.transform(df2.sub_category)

  #print('sub category features',vect_scat1.shape,':',vect_scat2.shape)

  

  

  vect_itype=Cvect.fit(df.item_type)

  vect_itype1=vect_itype.transform(df.item_type)

  vect_itype2=vect_itype.transform(df1.item_type)

  vect_itype3=vect_itype.transform(df2.item_type)

  #print('item type features',vect_itype1.shape,':',vect_itype2.shape)

  

  sp_mat=sp.hstack((vect_name1,vect_icond1,vect_brand1,vect_ship1,vect_desc1,vect_mcat1,vect_scat1,vect_itype1)).tocsr()

    

  sp_mat1=sp.hstack((vect_name2,vect_icond2,vect_brand2,vect_ship2,vect_desc2,vect_mcat2,vect_scat2,vect_itype2)).tocsr()

   

  sp_mat2=sp.hstack((vect_name3,vect_icond3,vect_brand3,vect_ship3,vect_desc3,vect_mcat3,vect_scat3,vect_itype3)).tocsr()

  

  return(sp_mat,sp_mat1,sp_mat2)









def get_rmsle(y_true, y_pred):

    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

%%time

df_category=pd.DataFrame(df_train.category.unique(),columns=['category'])

df_category['count']=df_category.category.apply(lambda x: len(df_train.category[df_train.category==x]))

df_category['rmsle']=0

df_category['skewness']=0

df_category['X_shape']=''

df_category['test_len']=0



df_predictions=pd.DataFrame(columns=['test_id','price'])

df_predictions



for i,cat in enumerate(df_category.category):

  try:

    df_cat=df_train[df_train.category==cat]

    df_cat=df_cat.drop(['category'],axis=1)

    

    df_cat_test=df_test[df_test.category==cat]

    df_cat_test=df_cat_test.drop(['category'],axis=1)

    



    df_category.iloc[i,3]=stat.skew(df_cat.price)

    

    df_cat_y=np.array(np.log1p(df_cat.price))

    df_cat_X=df_cat.drop(['price'],axis=1)

    

    df_pred=pd.DataFrame(df_cat_test.test_id,columns=['test_id','price'])

    df_cat_test=df_cat_test.drop(['test_id'],axis=1)



    for col in df_cat_test.columns.values:

      df_cat_test[col]=df_cat_test[col].astype('str')

    

    for col in df_cat_X.columns.values:

      df_cat_X[col]=df_cat_X[col].astype('str')

  

    df_cat_X=df_cat_X.drop(['train_id'],axis=1)

    

 



   



    j=0

    while True:

      

      j+=1

      if j==10:

        break

      try:

        train_X,test_X,train_y,test_y=train_test_split(df_cat_X,df_cat_y)

        print('In loop',i,'cat:',cat)

        if df_category.iloc[i,1]>=150000:

          sp_train,sp_test,sp_pred=get_sparse(train_X,test_X,df_cat_test)

          model = Ridge(alpha=4,solver='auto',fit_intercept=True,max_iter=1000,normalize=False)

        elif df_category.iloc[i,1]<150000 and df_category.iloc[i,1]>=100000:

          sp_train,sp_test,sp_pred=get_sparse(train_X,test_X,df_cat_test)

          model = Ridge(alpha=3.5,solver='auto',fit_intercept=True,max_iter=1000,normalize=False)

        elif df_category.iloc[i,1]<100000 and df_category.iloc[i,1]>=50000:

          sp_train,sp_test,sp_pred=get_sparse(train_X,test_X,df_cat_test)

          model = Ridge(alpha=2,solver='auto',fit_intercept=True,max_iter=1000,normalize=False)

        elif df_category.iloc[i,1]<50000 and df_category.iloc[i,1]>=10000:

          sp_train,sp_test,sp_pred=get_sparse(train_X,test_X,df_cat_test)

          model=Ridge(alpha=2,solver='auto',fit_intercept=True,max_iter=1000,normalize=False)

        else:

          sp_train,sp_test,sp_pred=get_sparse(train_X,test_X,df_cat_test)

          model=Ridge(solver='auto',fit_intercept=True,max_iter=1000,normalize=False)

      

        if j==8:

            if sp_train.shape[1]!=sp_test.shape[1]:

                dif=np.abs(sp_train.shape[1]-sp_test.shape[1])

                if sp_train.shape[1]>sp_test.shape[1]:

                    sp_test=sp.hstack((sp_test,sp_test[:,:dif])).tocsr()

                else:

                    sp_test=sp_test[:,dif:]    



        print('train:',sp_train.shape[1],'test:',sp_test.shape[1],'predict:',sp_pred.shape[1])

        

        

        

        

        if sp_train.shape[1]!=sp_pred.shape[1]:

          dif=np.abs(sp_train.shape[1]-sp_pred.shape[1])

          if sp_train.shape[1]>sp_pred.shape[1]:

            sp_pred=sp.hstack((sp_pred,sp_pred[:,:dif])).tocsr()

          else:

            sp_pred=sp_pred[:,dif:]

        print('train:',sp_train.shape[1],'test:',sp_test.shape[1],'predict:',sp_pred.shape[1])

    

        model.fit(sp_train,train_y)

        preds = model.predict(sp_test)

        df_pred['price']=model.predict(sp_pred)

    

        df_predictions=df_predictions.append(df_pred)

        

        break

      except:

        print('runtime error1')

        continue

        

    df_category.iloc[i,2]=get_rmsle(test_y,preds)

      

    df_category.iloc[i,5]=len(df_cat_test)

    

    if i==0:

      f_test_y=test_y

      f_preds=preds

    else:

      f_test_y=np.concatenate([f_test_y,test_y])

      f_preds=np.concatenate([f_preds,preds])



    df_category.iloc[i,4]=str(sp_train.shape)

    print('ridge score:',cat,'rmsle:', df_category.iloc[i,2],'len',len(df_cat_test))

    del df_cat, df_cat_X, train_X, test_X, train_y, test_y, sp_train, sp_test, preds,df_cat_test,df_pred

  except:

    print('runtime error2')

    continue







print(get_rmsle(f_test_y,f_preds))

df_predictions['price']=df_predictions.price.apply(lambda x: np.expm1(x))

print('Test set len:',len(df_test),'Predicted values len:',len(df_predictions.price),'difference:',len(df_test)-len(df_predictions.price))

df_predictions=df_predictions.sort_values(by='test_id').reset_index()

df_predictions=df_predictions.drop(['index'],axis=1)

df_predictions
df_predictions[df_predictions.test_id==535]
test[test.test_id==535]
df_category.sort_values(by='rmsle')
df_predictions.to_csv('ridge_submission.csv',index=False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from itertools import product

from sklearn import preprocessing

import xgboost as xgb

import gc

import pickle

from xgboost import plot_importance

import matplotlib.pyplot as plt

import os

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error





BASE_DIR= "../input/"
salesData= pd.read_csv(f"{BASE_DIR}sales_train.csv")

items= pd.read_csv(f"{BASE_DIR}items.csv")

testDF= pd.read_csv(f"{BASE_DIR}test.csv")

itemCategories= pd.read_csv(f"{BASE_DIR}item_categories.csv")

shops= pd.read_csv(f"{BASE_DIR}shops.csv")

salesData.drop(['date'],axis=1,inplace=True)
uniqueTestItems =  testDF['item_id'].nunique()

print("Number of unique test items = "+str(uniqueTestItems))



uniqueTrainItems =  salesData['item_id'].nunique()

print("Number of unique train items = "+str(uniqueTrainItems))
uniqueTestShops =  testDF['shop_id'].nunique()

print("Number of unique test shops = "+str(uniqueTestShops))



uniqueTrainShops =  salesData['shop_id'].nunique()

print("Number of unique train shops = "+str(uniqueTrainShops))
uniqueTrainItems = salesData['item_id'].unique()

itemsNotInTrainDF = testDF[ ~testDF['item_id'].isin( uniqueTrainItems ) ]

itemsInTrainDF = testDF[ testDF['item_id'].isin( uniqueTrainItems ) ]

itemsNotInTrainDF.shape[0]/itemsInTrainDF.shape[0]
uniqueTrainShops = salesData['shop_id'].unique()

shopsNotInTrainDF = testDF[ ~testDF['shop_id'].isin( uniqueTrainShops ) ]

shopsInTrainDF = testDF[ testDF['shop_id'].isin( uniqueTrainShops ) ]

shopsNotInTrainDF.shape[0]/shopsInTrainDF.shape[0]
filteredTrainDataWithItemAndShopPair= salesData[( salesData['item_id'].isin(testDF['item_id'].unique()) ) &

                            ( salesData['shop_id'].isin(testDF['shop_id'].unique()) )]

filteredTrainDataWithItemAndShopPair.shape[0]/salesData.shape[0]
filteredTrainDataWithItemPair= salesData[( salesData['item_id'].isin(testDF['item_id'].unique()) ) |

                            ( salesData['shop_id'].isin(testDF['shop_id'].unique()) )]

filteredTrainDataWithItemPair.shape[0]/salesData.shape[0]
salesData.describe().T
print(salesData[salesData['item_price']<=0].shape[0])

print(filteredTrainDataWithItemPair[filteredTrainDataWithItemPair['item_price']<=0].shape[0])
plt.subplots(figsize=(22, 8))

sns.boxplot(salesData['item_cnt_day'])

plt.show()
plt.subplots(figsize=(22, 8))

sns.boxplot(salesData['item_price'])

plt.show()
tmpCat= itemCategories.copy()

tmpCat['isHyphenInName']= tmpCat['item_category_name'].apply(lambda x: 1 if '-' in x else 0)

tmpCat['isHyphenInName'].sum()/tmpCat.shape[0]
tmpShops= shops.copy()

tmpShops['firstWordInName']= tmpShops['shop_name'].apply(lambda x: x.split(' ')[0])

tmpShops['firstWordInName'].nunique()/tmpShops.shape[0]
salesData= filteredTrainDataWithItemPair

salesData=salesData[ (salesData['item_price']<50000) & (salesData['item_cnt_day']<500)]
targetDF= salesData.groupby(['item_id','shop_id','date_block_num'],as_index=False).agg({'item_cnt_day':['sum'],'item_price':'mean'})

targetDF.columns= ['item_id','shop_id','date_block_num','target','avg_price']
%%time

cartesianList=[]

for i in range(34):

    tmpDF= targetDF[targetDF['date_block_num']==i]

    shopIDList= tmpDF['shop_id'].unique()

    itemIDList= tmpDF['item_id'].unique()

    cartesianRes = list(product(*[shopIDList, itemIDList, [i]]))

    cartesianList.append(cartesianRes)

cartesianList= np.vstack(cartesianList)

cartesianDF= pd.DataFrame(cartesianList,columns=['shop_id','item_id','date_block_num'])

del cartesianList
testDF['date_block_num'] = 34

testDF['date_block_num'] = testDF['date_block_num'].astype(np.int8)

testDF['shop_id'] = testDF['shop_id'].astype(np.int8)

testDF['item_id'] = testDF['item_id'].astype(np.int16)



cartesianDF = pd.concat([cartesianDF, testDF], ignore_index=True, sort=False, keys=['shop_id','item_id','date_block_num'])
%%time

resDF= cartesianDF.merge(targetDF,how='left',on=['shop_id','item_id','date_block_num'])
%%time

resDF['target'].fillna(0,inplace=True)

# just a tip from discussion forum -- https://mlwhiz.com/blog/2017/12/26/win_a_data_science_competition/

resDF['target']=resDF['target'] .clip(0,40)



# First get avg item price by month to fill empty places in price

itemPrice= salesData.groupby(['item_id','date_block_num'],as_index=False).agg({'item_price':'mean'})

itemPrice.columns=['item_id','date_block_num','avg_price_item']

resDF['avg_price'][ resDF['avg_price'].isna() ]= resDF[ resDF['avg_price'].isna() ].merge(itemPrice,how='left',on=['item_id','date_block_num'])['avg_price_item'].values 

resDF.describe().T



# To know what I did here; just analyze this small snippet

# a=pd.DataFrame({'x':['a','b','c'],'y':[1,2,np.nan]})

# b= pd.DataFrame({'x':['c','d'],'z':[65,7]})

# a['y'][ a['y'].isna() ]= a[ a['y'].isna() ].merge(b,on=['x'])['z'].values

# a
# Now fill the remaining price lags with avg item price

itemPrice= salesData.groupby(['item_id'],as_index=False).agg({'item_price':'mean'})

itemPrice.columns=['item_id','avg_price_item']

resDF['avg_price'][ resDF['avg_price'].isna() ]= resDF[ resDF['avg_price'].isna() ].merge(itemPrice,how='left',on=['item_id'])['avg_price_item'].values 
shops['shop_label']= shops['shop_name'].apply( lambda x: x.split(' ')[0])

le = preprocessing.LabelEncoder()

shops['shop_label']= le.fit_transform(shops['shop_label'])
itemCategories['cat_label']= itemCategories['item_category_name'].apply(lambda x: x.split('-')[0].strip())

itemCategories['cat_subLabel']= itemCategories['item_category_name'].apply(lambda x: x.split('-')[1].strip() if '-' in x else x.split('-')[0].strip())



le = preprocessing.LabelEncoder()

itemCategories['cat_label']= le.fit_transform(itemCategories['cat_label'])

itemCategories['cat_subLabel']= le.fit_transform(itemCategories['cat_subLabel'])
resDF= resDF.merge(items[['item_id','item_category_id']],on=['item_id'],how='left')

resDF= resDF.merge(itemCategories[['cat_label','cat_subLabel','item_category_id']],how='left',on=['item_category_id'])

resDF= resDF.merge(shops[['shop_label','shop_id']],how='left',on=['shop_id'])
meanEncodingCols =[

    ( ['shop_id'],[1,2,3,6,12]),

    ( ['item_id'],[1,2,3,6,12]),

    ( ['item_category_id'],[1,2,3,6,12]),

    (['item_id','shop_id'],[1,2,3,6,12]),

    (['item_category_id','shop_id'],[1,2,3,6,12]),

    (['cat_label'],[1]),

    (['cat_label','item_id'],[1,2,3]),

    (['cat_subLabel'],[1]),

    (['shop_label'],[1])

]
def generateGroupByFeatures(ipDF,groupCols,aggDict):

    groupByAlias = {'item_id':'item','shop_id':'shop','item_category_id':'item_cat','date_block_num':'date',

                    'month':'month','target':'target','cat_label':'cat_label','cat_subLabel':'cat_subLabel',

                   'shop_label':'shop_label'}

    aggColAlias= {'item_cnt_day':'cnt','item_price':'price','target':'target','avg_price':'price'}

    

    grpByName= '_'.join( [groupByAlias[x] for x in groupCols] )+'_'

    colNameList=[]

    

    for aggCol,aggFnArr in aggDict.items():

        for f in aggFnArr:

            fName = f.__name__ if hasattr(f, '__call__') else f

            colNameList.append( grpByName+aggColAlias[aggCol]+"_"+fName )

    returnDF= ipDF.groupby(groupCols,as_index=False).agg(aggDict)

    returnDF.columns= groupCols+colNameList

    return returnDF
for c in resDF.columns:

    if 'int' in str(resDF[c].dtype):

        resDF[c]= resDF[c].astype('Int32')

        if resDF[c].max()<32000:

            resDF[c]= resDF[c].astype('Int16')

        if resDF[c].max()<120:

            resDF[c]= resDF[c].astype('Int8')

    if 'float' in str(resDF[c].dtype):

        resDF[c]= resDF[c].astype('Float32')

        if resDF[c].max()<65000:

            resDF[c]= resDF[c].astype('Float16')
grpByItem= generateGroupByFeatures(salesData,['item_id'],{'item_price':['min','max',np.var]})

grpByItem['historic_price_shift']= (grpByItem['item_price_max']-grpByItem['item_price_min'])/grpByItem['item_price_max']

grpByItem.drop(['item_price_max','item_price_min'],axis=1,inplace=True)

resDF= resDF.merge(grpByItem,how='left',on=['item_id'])
%%time

for m in meanEncodingCols:

    grpCols= m[0]

    lagArr= m[1]

    

    tmpDF= resDF.copy()

    grpCols= grpCols+['date_block_num']

    groupDF= generateGroupByFeatures(tmpDF,grpCols,{'target':['mean']})

    tmpDF= tmpDF.merge(groupDF,how='left',on= grpCols)

    

    # mean encoded column

    meanEncodedColName= [c for c in groupDF.columns if 'target_mean' in c]

    assert len(meanEncodedColName)==1

    meanEncodedColName= meanEncodedColName[0]

    

    for lag in lagArr:

        shiftedDF= tmpDF[['item_id','shop_id','date_block_num',meanEncodedColName]].copy()

        shiftedDF['date_block_num']= shiftedDF['date_block_num']+lag

        

        lagColName= meanEncodedColName+"_lag_"+str(lag)

        shiftedDF.rename({meanEncodedColName:lagColName},axis=1,inplace=True)

        

        resDF= resDF.merge(shiftedDF,how='left',on=['item_id','shop_id','date_block_num'])

        del shiftedDF

    del tmpDF
%%time

for lag in [1,2,3]:

    tmpDF= resDF[['item_id','date_block_num','avg_price']].copy()

    tmpDF['date_block_num']+= lag

    tmpGrpByItem= tmpDF.groupby(['item_id','date_block_num'],as_index=False).agg({'avg_price':['mean']})

    tmpGrpByItem.columns= ['item_id','date_block_num','item_price_lag'+str(lag)]

    

    resDF= resDF.merge(tmpGrpByItem,how='left',on=['item_id','date_block_num'])

    del tmpGrpByItem

    del tmpDF
%%time

for lag in [1,2,3]:

    tmpDF= resDF[['item_category_id','date_block_num','avg_price']].copy()

    tmpDF['date_block_num']+= lag

    tmpGrpByItem= tmpDF.groupby(['item_category_id','date_block_num'],as_index=False).agg({'avg_price':['mean']})

    tmpGrpByItem.columns= ['item_category_id','date_block_num','item_cat_price_lag'+str(lag)]

    

    resDF= resDF.merge(tmpGrpByItem,how='left',on=['item_category_id','date_block_num'])

    del tmpGrpByItem

    del tmpDF
for col in resDF.columns:

    if 'price' not in col and ( resDF[col].isnull().any() ):

        resDF[col].fillna(0, inplace=True)
for col in resDF.columns:

    if 'price' in col:

        resDF[col]= resDF[col].fillna(resDF[col].median())
gc.collect()

resDF.to_pickle('data.pkl')
resDF['month']= resDF['date_block_num']%12

resDF=resDF[ resDF['date_block_num']>12 ]
for c in ['item_price_lag1', 'item_price_lag2','item_price_lag3']:

    resDF[c+'_diff']= resDF[c]/resDF['avg_price']

    resDF.drop([c],axis=1,inplace=True)
resDF.describe().T
%%time

X= resDF[resDF['date_block_num']<33]

Y= X['target']

X.drop(['target'],axis=1,inplace=True)



X_val= resDF[(resDF['date_block_num']==33)&(resDF['target']==0)]

Y_val= X_val['target']

X_val.drop(['target'],axis=1,inplace=True)



X_val_1= resDF[(resDF['date_block_num']==33)&(resDF['target']!=0)]

Y_val_1= X_val_1['target']

X_val_1.drop(['target'],axis=1,inplace=True)





model = xgb.XGBRegressor(max_depth=5 ,

    n_estimators=100,

    min_child_weight=20, 

    colsample_bytree=0.6,

    seed=42,

    n_jobs=-1)

model.fit(X,Y,eval_metric="rmse", 

          eval_set=[(X,Y),(X_val, Y_val),(X_val_1, Y_val_1)],

          verbose=20,

         early_stopping_rounds=10)

pickle.dump(model, open("xgb.pickle.dat", "wb"))

plt.rcParams["figure.figsize"] = (15, 30)

plot_importance(model)

plt.show()
%%time



X= resDF[resDF['date_block_num']<33]

Y= X['target']

X.drop(['target'],axis=1,inplace=True)



X_val= resDF[(resDF['date_block_num']==33)]

Y_val= X_val['target']

X_val.drop(['target'],axis=1,inplace=True)



lr = LinearRegression(n_jobs=-1)



lr.fit(X.values, Y.values)

pickle.dump(lr, open("lr.pickle.dat", "wb"))
X_train= resDF[resDF['date_block_num']<33].drop(['target'],axis=1)

X_test= resDF[resDF['date_block_num']==34].drop(['target'],axis=1)

X_val= resDF[resDF['date_block_num']==33].drop(['target'],axis=1)



Y_train= resDF[resDF['date_block_num']<33]['target']

Y_val= resDF[resDF['date_block_num']==33]['target']



xgbModel= model;lrModel= lr;
meta_xg_train= xgbModel.predict(X_train)

meta_lr_train= lrModel.predict(X_train)



meta_xg_val= xgbModel.predict(X_val)

meta_lr_val= lrModel.predict(X_val)



meta_xg_test= xgbModel.predict(X_test)

meta_lr_test= lrModel.predict(X_test)



metaTrain= np.c_[meta_xg_train,meta_lr_train]

print(metaTrain.shape,Y_train.shape)



metaVal= np.c_[meta_xg_val,meta_lr_val]

print(metaVal.shape,Y_val.shape)



metaTest= np.c_[meta_xg_test,meta_lr_test]

print(metaTest.shape)
lr= LinearRegression(n_jobs=-1)

lr.fit(metaTrain,Y_train.values)



metaVal= np.c_[meta_xg_val,meta_lr_val]

Y_train_pred= lr.predict(metaTrain)

Y_val_pred= lr.predict(metaVal)





print(np.sqrt(mean_squared_error(Y_val_pred,Y_val.values)))

print(np.sqrt(mean_squared_error(Y_train_pred,Y_train.values)))
drop_cols= [ 'target']

testDF = resDF[ resDF['date_block_num']==34]

testDF['target']= lr.predict(metaTest)



submission = pd.DataFrame({

    "ID": testDF['ID'], 

    "item_cnt_month": testDF['target'].clip(0,20)

})

submission['ID']= submission['ID'].astype(np.int32)

submission.to_csv('submission.csv', index=False)
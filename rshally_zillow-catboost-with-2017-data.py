import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import gc

import datetime as dt

from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import MiniBatchKMeans

from catboost import CatBoostRegressor



myfolder = '../input/'

print('loading files...')



prop = pd.read_csv(myfolder+'properties_2017.csv',low_memory=False)

prop.rename(columns={'parcelid': 'ParcelId'}, inplace=True)   # make it the same as sample_submission

train = pd.read_csv(myfolder+'train_2016_v2.csv')

train.rename(columns={'parcelid': 'ParcelId'},inplace=True)

sample = pd.read_csv(myfolder+'sample_submission.csv')

print(train.shape, prop.shape, sample.shape)

train17 = pd.read_csv(myfolder+'train_2017.csv')

train17.rename(columns={'parcelid': 'ParcelId'},inplace=True)

print(train17.shape)

train=pd.concat([train,train17])

del train17

print(train.shape)
print('preprocessing, fillna, dtypes ...')



prop['longitude']=prop['longitude'].fillna(prop['longitude'].median()) / 1e6   #  convert to float32 later

prop['latitude'].fillna(prop['latitude'].median()) / 1e6

prop['censustractandblock'].fillna(prop['censustractandblock'].median()) / 1e12

train = train[train['logerror'] <  train['logerror'].quantile(0.9975)]  # exclude 0.5% of outliers

train = train[train['logerror'] >  train['logerror'].quantile(0.0025)]



print('qualitative ...')

qualitative = [f for f in prop.columns if prop.dtypes[f] == object]

prop[qualitative] = prop[qualitative].fillna('Missing')

for c in qualitative:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values)).astype(int)



print('smallval ...')

smallval = [f for f in prop.columns if np.abs(prop[f].max())<100]

prop[smallval] = prop[smallval].fillna('Missing')

for c in smallval:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values)).astype(np.int8)



print('other ...')

other=['regionidcounty','fips','propertycountylandusecode','propertyzoningdesc','propertylandusetypeid']

prop[other] = prop[other].fillna('Missing')

for c in other:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values)).astype(int)



randomyears=pd.Series(np.random.choice(prop['yearbuilt'].dropna().values,len(prop)))

prop['yearbuilt']=prop['yearbuilt'].fillna(randomyears).astype(int)

med_yr=prop['yearbuilt'].quantile(0.5)

prop['New']=prop['yearbuilt'].apply(lambda x: 1 if x > med_yr else 0).astype(np.int8)  # adding a new feature



prop['unitcnt'] = prop['unitcnt'].fillna(1).astype(int)    

prop['Condo']=prop['unitcnt'].apply(lambda x: 1 if x > 1 else 0).astype(np.int8)    # adding a new feature

    

feat_to_drop=[ 'finishedsquarefeet50', 'finishedfloor1squarefeet', 'finishedsquarefeet15', 

              'finishedsquarefeet13','assessmentyear']

prop.drop(feat_to_drop,axis=1,inplace=True)   # remove because too many missing or irrelevant



prop['lotsizesquarefeet'].fillna(prop['lotsizesquarefeet'].quantile(0.001),inplace=True)

prop['finishedsquarefeet12'].fillna(prop['finishedsquarefeet12'].quantile(0.001),inplace=True)

prop['calculatedfinishedsquarefeet'].fillna(prop['finishedsquarefeet12'],inplace=True)

prop['taxamount'].fillna(prop['taxamount'].quantile(0.001),inplace=True)

prop['landtaxvaluedollarcnt'].fillna(prop['landtaxvaluedollarcnt'].quantile(0.001),inplace=True)

prop.fillna(0,inplace=True)

    

print('quantitative ...')   

quantitative = [f for f in prop.columns if prop.dtypes[f] == np.float64]

prop[quantitative] = prop[quantitative].astype(np.float32) 



cfeatures = list(prop.select_dtypes(include = ['int64', 'int32', 'uint8', 'int8']).columns)

for c in qualitative:  prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values))



# some quantitative features have a limited number of values (eg ZIP code)    

for c in ['rawcensustractandblock',  'regionidcity',  'regionidneighborhood',  'regionidzip',  'censustractandblock'] :

    prop[c] = LabelEncoder().fit(list(prop[c].values)).transform(list(prop[c].values))



gc.collect()
print('create new features and the final dataframes ...')



#replace latitudes and longitudes with 500 clusters  (similar to ZIP codes)

coords = np.vstack(prop[['latitude', 'longitude']].values)

sample_ind = np.random.permutation(len(coords))[:1000000]

kmeans = MiniBatchKMeans(n_clusters=500, batch_size=100000).fit(coords[sample_ind])

prop['Cluster'] = kmeans.predict(prop[['latitude', 'longitude']])



prop['Living_area_prop'] = prop['calculatedfinishedsquarefeet'] / prop['lotsizesquarefeet']

prop['Value_ratio'] = prop['taxvaluedollarcnt'] / prop['taxamount']

prop['Value_prop'] = prop['structuretaxvaluedollarcnt'] / prop['landtaxvaluedollarcnt']

prop['Value_prop'].fillna(0,inplace=True)

prop['Taxpersqrtfoot']=prop['taxamount'] / prop['finishedsquarefeet12']



train['transactiondate'] = pd.to_datetime(train.transactiondate)

train['Month'] = train['transactiondate'].dt.month.astype(np.int8)

train['Day'] = train['transactiondate'].dt.day.astype(np.int8)

train['Season'] = train['Month'].apply(lambda x: 1 if x in [1,2,9,10,11,12] else 0).astype(np.int8)



month_err=(train.groupby('Month').aggregate({'logerror': lambda x: np.mean(x)})- train['logerror'].mean()).values

train['Meanerror']=train['Month'].apply(lambda x: month_err[x-1]).astype(np.float32)



train['abserror']=train['logerror'].abs()

month_abs_err=(train.groupby('Month').aggregate({'abserror': lambda x: np.mean(x)})- train['abserror'].mean()).values

train['Meanabserror']=train['Month'].apply(lambda x: month_abs_err[x-1]).astype(np.float32)

train.drop(['abserror'], axis=1,inplace=True)



for c in ['Meanerror','Meanabserror']: train[c]=LabelEncoder().fit(list(train[c].values)).transform(list(train[c].values))

for c in ['Meanerror','Meanabserror']: train[c]=train[c].astype(np.int8)



print(prop.shape, train.shape)

gc.collect()
# define X,y that can be used either as the training set or be split into train and eval sets (combine prop and train)

# and define X_test for the final prediction to submit (combine prop and sample)



X = train.merge(prop, how='left', on='ParcelId')

y = X['logerror']

X.drop(['ParcelId', 'logerror', 'transactiondate'], axis=1,inplace=True)



features=list(X.columns)

cfeatures = list(X.select_dtypes(include = ['int64', 'int32', 'uint8', 'int8']).columns)



X_test = (sample.merge(prop, on='ParcelId', how='left')).loc[:,features]

X_test['Season']=np.int8(1)

X_test['Day']=np.int8(15)

X_test['Month']=np.int8(10) 

X_test['Meanerror']=np.int8(10)

X_test['Meanabserror']=np.int8(10)



print(X.shape, y.shape, X_test.shape)

del prop, train

gc.collect()
print('catboost training ...')

X_train, X_eval, y_train, y_eval = train_test_split(X,y, test_size=0.15, random_state=1)

model = CatBoostRegressor(iterations=1000,learning_rate=0.002, depth=7, loss_function='MAE', 

                          eval_metric='MAE', random_seed=1)

model.fit(X_train, y_train, eval_set=(X_eval, y_eval), use_best_model=True, verbose=False, plot=True)

pred1 = model.predict(X_train)

pred2 = model.predict(X_eval)

print(' catboost MAE train  {:.4f}'.format(np.mean(np.abs(y_train.values-pred1) )))

print(' catboost MAE eval   {:.4f}'.format(np.mean(np.abs(y_eval.values-pred2) )))

del pred1, pred2

gc.collect()
FeatImp=pd.DataFrame(model.feature_importances_, index=features, columns=['Importance'])

FeatImp=FeatImp.sort_values('Importance')

FeatImp.plot(kind='barh', figsize=(8,14))

plt.show()
print('catboost predict and submit ...')



for month in [ 10,11,12]:

    print('month ',month)

    X_test['Month']=np.int8(month) 

    X_test['Meanerror']=X['Meanerror'].loc[X['Month']==month].mean().astype(np.int8)

    X_test['Meanabserror']=X['Meanerror'].loc[X['Month']==month].mean().astype(np.int8)

    pred = model.predict(X_test)

    sample['2016' + str(month)] = pred*1.05

    sample['2017' + str(month)] = pred

    print(' catboost MAE {}  {:.4f}'.format(month,np.mean(np.abs(sample['2017' + str(month)]-0) )))



sample.to_csv('submission_cat1.csv', index = False, float_format = '%.5f')

gc.collect()
import numpy as np
import pandas as pd
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
print('Loading datasets...')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

y = np.log1p( train['target'].values )
IDtest  = test['ID'].values

print('Merging all...')
test['target'] = np.nan
train = train.append(test).reset_index() # merge train and test
del test
gc.collect()

print("Create Model...")
train = train[train.columns.drop(['index','ID','target'])] # only get "X" vector
gc.collect()
%%time
print("rounding...")
for i in train.columns:
    train[i] = np.round( np.log1p(train[i]) , decimals=3 )

gc.collect()
print(train.head(5))
%%time
print("To String...")

tmp = train[train.columns[0]].apply(str)
tmp[ tmp=='0.0' ] = ''
CV = pd.DataFrame()
CV['features'] = tmp

for feat in train.columns[1:]:
    tmp = train[feat].apply(str)
    tmp[ tmp=='0.0' ] = ''
    CV['features'] = CV['features'] + tmp + ' '

del train
gc.collect()
print( CV )
%%time
rd = CountVectorizer( lowercase=True, ngram_range=(1, 1), max_df=0.99, min_df=2)
train = rd.fit_transform( CV['features'] )
del rd, CV
gc.collect()
print(train.shape)
%%time
rd = RandomForestRegressor(n_estimators=2222, criterion='mse', max_depth=10, max_features=0.51, n_jobs=-1)
rd.fit( train[:4459,:], y )
gc.collect()
%%time
sub = pd.DataFrame( {'ID':IDtest} )
sub['target'] = np.expm1( rd.predict( train[4459:,:] ) ).astype(np.int)
sub.to_csv( 'giba-rf-1.csv', index=False )
print( sub.head(20) )
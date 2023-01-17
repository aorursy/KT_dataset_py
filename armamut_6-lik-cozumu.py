import numpy as np

import pandas as pd

pd.options.display.max_rows = 1000

pd.options.display.max_columns = 1000



from collections import Counter

%matplotlib inline



import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode()



import gc
# Daily transaction file



# productid - unique id for each product (int32)

# date - date of product action

# soldquantity - sales of product (int16)

# stock - beginning stock count of product (int32)

# clickcount - # of clicks of product (int32)

# favoredcount - # of favored click of product (int16)



df = pd.read_csv('../input/dailyProductActions.csv', parse_dates=['date'], dtype={'productid': 'int32'})

df['soldquantity'] = df['soldquantity'].fillna(0).astype(np.int16)

#df['stock'] = df['stock'].fillna(0).astype(np.int32)

df['clickcount'] = df['clickcount'].fillna(0).astype(np.int32)

df['favoredcount'] = df['favoredcount'].fillna(0).astype(np.int16)

df = df.sort_values(by=['productid', 'date']).reset_index(drop=True)

print(df.shape)

display(df.head())
# Submission file

submission = pd.read_csv('../input/SampleSubmission.csv')

print(submission.shape)

display(submission.head())
# Products file

products = pd.read_csv('../input/product.csv').sort_values(by='productid')

print(products.shape)

display(products.head())
date_start = pd.to_datetime('2018-11-01')

date_end = pd.to_datetime('2019-01-11')

date_days = int((date_end - date_start) / pd.to_timedelta('1d')) + 1

print(date_days)
df.groupby('date')['soldquantity'].mean().plot.bar(figsize=(16, 4))
#

# 9, 10, 11 Kasım İndirimi

# https://www.trendyol.com/s/11-kasim-indirim-gunleri

#

# 20-25 Kasım Efsane Günler

# https://www.trendyol.com/s/black-friday

# Black Friday, November 29

#

# 20 - 21 Aralık

# https://www.trendyol.com/s/21-aralik-en-uzun-gece

#

# 2019-01-05 from observation ?

#



indirim_gunleri = [

    '2018-11-09', '2018-11-10', '2018-11-11', '2018-11-20', '2018-11-21',

    '2018-11-22', '2018-11-23', '2018-11-24', '2018-12-20', '2018-12-21',

    #'2019-01-05'

]

indirim_gunleri = [pd.to_datetime(c) for c in indirim_gunleri]

df_indirim = pd.DataFrame(indirim_gunleri, columns=['date'])

df_indirim['indirim'] = 1

df_indirim = df_indirim.set_index('date')

df_indirim = df_indirim.reindex(pd.date_range(date_start, date_end), fill_value=0).reset_index()

df_indirim = df_indirim.rename(columns={'index':'date'})

df_indirim['indirim_next7days'] = df_indirim['indirim'].rolling(7).sum().shift(-7).fillna(0) / 7

df_indirim['indirim_prev7days'] = df_indirim['indirim'].rolling(7).sum().fillna(0) / 7

#for i in range(1, 8):

#    df_indirim[f'indirim_next_{i}'] = df_indirim['indirim'].shift(-i).fillna(0).astype(int)

df_indirim.head(10)
print('gender = ', products.gender.max())

print('color = ', products.color.max())

print('categoryid = ', products.categoryid.max())

print('brandid = ', products.brandid.max())

print('subcategoryid = ', products.subcategoryid.max())
# Generate expanded transaction matrix

productids = sorted(set(submission.productid.unique()) | 

                    set(df.productid.sample(10000)))

#productids = sorted(set(df.productid.unique()))

print(len(productids))

tmp = df[df.productid.isin(set(productids))]

print(tmp.shape)

tmp2 = pd.DataFrame({

    'productid': np.repeat(productids, date_days),

    'date': np.tile(pd.date_range(date_start, date_end), len(productids)),

})

tmp = tmp2.merge(tmp, on=['productid', 'date'], how='left')

del tmp2

tmp['stock'] = tmp.groupby('productid')['stock'].apply(lambda x:x.fillna(method='ffill').fillna(method='bfill')) #.clip(0,1000) / 1000

tmp = tmp.fillna(0)

print(tmp.shape)

tmp.head(30)
tmp = tmp.merge(tmp.groupby('date')['clickcount'].mean().to_frame('daily_clickcount').reset_index(),

               on='date',

               how='left')

tmp['clickratio'] = tmp['clickcount'] / tmp['daily_clickcount']

tmp.head()
tmp = tmp.merge(tmp.groupby('date')['favoredcount'].mean().to_frame('daily_favoredcount').reset_index(),

               on='date',

               how='left')

tmp['favoredratio'] = tmp['favoredcount'] / tmp['daily_favoredcount']

tmp.head()
tmp = tmp.merge(tmp.groupby('date')['soldquantity'].mean().to_frame('daily_soldquantity').reset_index(),

               on='date',

               how='left')

tmp['soldratio'] = tmp['soldquantity'] / tmp['daily_soldquantity']

tmp.head()
tmp = tmp.merge(tmp.groupby('date')['stock'].mean().to_frame('daily_stock').reset_index(),

               on='date',

               how='left')

tmp['stockratio'] = tmp['stock'] / tmp['daily_stock']

tmp.head()
# Define Target



# Target hesaplamak için 7 gün kaydırıyoruz. Ancak, bütün ürünleri 7 kere

# kaydırdığımız için, bir sonraki ürünün ilk 7 gün toplamı, bir önceki ürünün son target'ıymış gibi geliyor.

# Bunu önlemek için her ürünün en son 73. değerini siliyoruz.

rs = tmp['soldquantity'].fillna(0).copy()

rss = []

for i in range(1, 8):

    rs = rs.shift(-1)

    rs[71::72] = np.nan

    rss.append(rs)

tmp[f'target'] = np.sum(rss, axis=0)

tmp.head()
for alpha in [0.2, 0.1]:

    for col in ['soldquantity', 'stock', 'clickcount', 'favoredcount',

               'clickratio', 'favoredratio', 'soldratio', 'stockratio']:

        print(f'ewm_{alpha}_{col}')

        tmp[f'ewm_{alpha}_{col}'] = tmp.groupby('productid')[col].apply(lambda x:x.ewm(alpha=alpha).mean())

tmp['ewm_0.2_fc'] = tmp['ewm_0.2_favoredcount'] / tmp['ewm_0.2_clickcount']

tmp['ewm_0.2_sc'] = tmp['ewm_0.2_soldquantity'] / tmp['ewm_0.2_clickcount']

tmp['ewm_0.2_sf'] = tmp['ewm_0.2_soldquantity'] / tmp['ewm_0.2_favoredcount']



tmp.head()
tmp = tmp.merge(df_indirim, on='date', how='left')

tmp = tmp.merge(products, on='productid', how='left')

tmp = tmp.sort_values(by=['productid', 'date'])

print(tmp.shape)

tmp.head()
submissionids = sorted(set(submission.productid.unique()))



prediction_days = 7

date_predict = pd.to_datetime('2019-01-11') - pd.to_timedelta(f'{prediction_days}d')

date_train   = date_predict - pd.to_timedelta('8d')

date_test    = date_predict - pd.to_timedelta('1d')

print(date_train)

print(date_test)



df_train   = tmp[(tmp.date <= date_train)]

                 #(tmp[f'indirim_next7days'] == 0) &

                 #(tmp.productid.isin(set(submissionids)))]



df_test    = tmp[(tmp.date > date_train) &

                 (tmp.date <= date_test) &

                 (tmp.productid.isin(set(submissionids)))]



df_val     = tmp[(tmp.date > date_test) &

                 (tmp.date <=  date_predict) &

                 (tmp.productid.isin(set(submissionids)))]



#df_predict = tmp[(tmp.productid.isin(set(submissionids))) &

#                 (tmp.target.isnull())]



print(f'df_train.shape   : {df_train.shape}')

print(f'df_test.shape    : {df_test.shape}')

print(f'df_val.shape     : {df_val.shape}')

#print(f'df_predict.shape : {df_predict.shape}')
col_target = f'target'

col_not_use = ['productid', 'date', 'target']

col_not_use += ['ewm_soldratio_trend']

col_not_use += [c for c in tmp.columns if c.startswith('target')]

#col_not_use += [c for c in tmp.columns if c.startswith('daily')]

#col_not_use += [c for c in tmp.columns if c.endswith('stockratio')]

#col_not_use += [c for c in tmp.columns if 'stockratio' in c]

#col_not_use += [c for c in tmp.columns if c.startswith('stat_productid')]

#col_not_use += [c for c in tmp.columns if c.startswith('ewm_0.1_stat')]



col_use = [c for c in tmp.columns if c not in col_not_use]

col_cat = ['gender', 'color', 'categoryid', 'brandid', 'subcategoryid']

#col_cat = [c for c in col_cat if c in col_use]



print('col_use', col_use)

print('col_target',col_target)



from lightgbm import LGBMRegressor

#models = dict()

#for col_target in col_targets:



models = []

for random_state in [42,84,1,2,3]:

    print(f'Model for {col_target}')



    model = LGBMRegressor(objective='regression',

                          num_leaves=31,

                          random_state=random_state,

                          subsample=0.8,

                          colsample_bytree=0.8,

                          learning_rate=0.05,

                          n_estimators=2000,

                          reg_alpha=5,

                          reg_lambda=5,)

    print(model)

    model.fit(df_train[col_use], df_train[col_target],

              categorical_feature=col_cat,

              early_stopping_rounds=100,

              eval_set=(df_test[col_use], df_test[col_target]),

              eval_metric=['rmse'],

              verbose=100)

    col_pred = col_target.replace('target', 'pred')

    #models[col_pred] = model

    from sklearn.metrics import mean_squared_error

    preds = model.predict(df_test[col_use])

    print('test rmse:', np.sqrt(mean_squared_error(preds.clip(0), df_test[col_target])))



    preds = model.predict(df_val[col_use])

    print('val rmse:', np.sqrt(mean_squared_error(preds.clip(0), df_val[col_target])))

    

    models.append(model)



print('OK')

print('Done')
import eli5

eli5.explain_weights_lightgbm(model, top=500)
# subm 15 : val rmse: 7.884177554137836

# subm 16 : val rmse: 7.752352602499219

preds = []

for model in models:

    preds.append(model.predict(df_val[col_use]))



print('val rmse:', np.sqrt(mean_squared_error(pd.DataFrame(np.array(preds)).median(axis=0).clip(0), df_val[col_target])))

# subm 15 : val rmse: 7.884177554137836

# subm 16 : val rmse: 7.752352602499219

preds = model.predict(df_val[col_use])

print('val rmse:', np.sqrt(mean_squared_error(preds.clip(0), df_val[col_target])))

#print(np.sqrt(mean_squared_error(model.predict(df_test[col_use]), df_test[col_target])))

for i in range(-30, -6):

    dt = pd.to_datetime('2019-01-11') + pd.to_timedelta(f'{i}D')

    df_tmp = tmp[(tmp.date == dt)]

    df_tmp = df_tmp[(df_tmp.productid.isin(set(submissionids)))]

    print(i, dt, np.sqrt(mean_squared_error(model.predict(df_tmp[col_use].clip(0)), df_tmp[col_target])))
py.iplot([

    go.Scattergl(x=tmp[tmp.date == pd.to_datetime('2019-01-01')][col_target], y=model.predict(tmp[tmp.date == pd.to_datetime('2019-01-01')][col_use]), mode='markers'),

    go.Scattergl(x=tmp[tmp.date == pd.to_datetime('2019-01-02')][col_target], y=model.predict(tmp[tmp.date == pd.to_datetime('2019-01-02')][col_use]), mode='markers'),

    go.Scattergl(x=tmp[tmp.date == pd.to_datetime('2019-01-03')][col_target], y=model.predict(tmp[tmp.date == pd.to_datetime('2019-01-03')][col_use]), mode='markers'),

    go.Scattergl(x=tmp[tmp.date == pd.to_datetime('2019-01-04')][col_target], y=model.predict(tmp[tmp.date == pd.to_datetime('2019-01-04')][col_use]), mode='markers')

])
preds = dict()

for i in range(1, 8):

    dt = pd.to_datetime('2019-01-04') + pd.to_timedelta(f'{i}D')

    print(f'predicting {i}, {dt}')

    df_predict = tmp[tmp.productid.isin(set(submissionids))]

    df_predict = df_predict[df_predict.date == dt]

    #display(df_predict.head())

    preds[i] = model.predict(df_predict[col_use])

print(len(preds))
# Genel bakış.

pd.DataFrame(preds).clip(0)
df_subm = pd.DataFrame()

df_subm['productid'] = df_predict['productid']

#df_subm['sales'] = preds[7].clip(0)

df_subm['sales'] = pd.DataFrame(preds).max(axis=1).values

df_subm.head()
df_subm.to_csv('../submission_16.csv', index=False)
# Kategori içerisinde bir ürün diğer ürünlere göre tercih ediliyor olabilir.

# Bunun belirleyicisi fiyat, renk ya da başka birşey olabilir.

# Belirli bir renk, belirli bir marka diğerlerine göre daha çok tercih ediliyor olabilir.

# Stokta kalmadı hikayesi de önemli olabilir.

# Bazı ürünlerin çok satışı yok. Onları elemek gerekebilir.

# Bazı günlerde satışlar anormal devam ediyor. Onları elemek gerekebilir. Onun dışında düzenli gidiyor mesela.

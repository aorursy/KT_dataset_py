import time

import datetime



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



from sklearn import model_selection, preprocessing

import xgboost as xgb
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')
train.info()
test.info()
macro.info()
id_test = test['id']



valid_split = int(0.8*train.shape[0])

valid = train.ix[valid_split+1:]

train = train.ix[:valid_split]



# train = pd.concat((train,valid)) # use for final model
n_train = len(train)

n_valid = len(valid)

n_test = len(test)

n_total = n_train + n_valid + n_test

    

print("训练集样本数目为：", n_train)

print("测试集样本数目为：", n_valid)

print("验证集样本数目为：", n_test,'\n')

    

print('训练集样本所占比例为：',np.round(n_train/(n_train+n_valid+n_test),2))

print('测试集样本所占比例为：',np.round(n_valid/(n_train+n_valid+n_test),2))

print('验证集样本所占比例为：',np.round(n_test/(n_train+n_valid+n_test),2))

print('\n')



print ('训练样本的日期跨度为:',train.iloc[0,1] + '  到  ' + train.iloc[-1,1])

print ('测试样本的日期跨度为:',valid.iloc[0,1] + '  到  ' + valid.iloc[-1,1])

print ('验证样本的日期跨度为:',test.iloc[0,1] + '  到  ' + test.iloc[-1,1])
price = train['price_doc']

log_price = np.log1p(price)

plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

#plt.xlabel('index',fontsize=25)

plt.ylabel('price',fontsize=25)

plt.scatter(range(len(train)), np.sort(price))

plt.subplot(1,2,2)

#plt.xlabel('index',fontsize=25)

plt.ylabel('log price',fontsize=25)

plt.scatter(range(len(train)), np.sort(log_price))

plt.show()



plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

plt.title('price', fontsize=25)

hist = plt.hist(price, bins=64)[0]

plt.subplot(1,2,2)

plt.title('log price', fontsize=25)

hist = plt.hist(log_price, bins=64)[0]

plt.show()
def moving_average(a, n=3) :

    ret = np.cumsum(a, dtype=float)

    ret[n:] = ret[n:] - ret[:-n]

    return ret[n - 1:] / n





plt.figure(figsize=(18,6))

plt.subplot(1,2,1)

plt.tick_params(labelsize=14)

unique = train.pivot_table(index=['timestamp'],aggfunc='count')

plt.ylim(0,80)

plt.title('number of house sales by date',fontsize=20)

plt.plot(unique['price_doc'].values)

plt.plot(moving_average(unique['price_doc'].values,30),lw=3,color='r')



plt.subplot(1,2,2)

plt.tick_params(labelsize=14)

unique = train.pivot_table(index=['timestamp'],aggfunc='median')

plt.ylim(0.4e7,0.8e7)

plt.title('median house sales by date',fontsize=20)

plt.plot(unique['price_doc'].values)

plt.plot(moving_average(unique['price_doc'].values,30),lw=3,color='r')

print('宏观经济学特征数目:',macro.shape[0])

print('有缺失值的列的数目:',train.isnull().any().sum(),'/',len(train.columns))

print('有缺失值的行的数目为:',train.isnull().any(axis=1).sum(),'/',len(train))





ranking = train.loc[:,train.isnull().any()].isnull().sum().sort_values()

x = ranking.values / len(train)

index = np.arange(len(ranking))

    

plt.bar(index, x)

plt.xlim(0,50)

plt.xlabel('Features')

plt.ylabel('percent of NaN observations')

plt.title('percent of null data points for each feature')

plt.show()
macro_feat = ['oil_urals','usdrub','brent','rts','micex','micex_rgbi_tr','deposits_rate','mortgage_growth','mortgage_rate',

             'rent_price_4+room_bus','rent_price_3room_bus','rent_price_2room_bus','rent_price_1room_bus',

             'rent_price_3room_eco','rent_price_2room_eco','rent_price_1room_eco']



df = macro[['timestamp'] + macro_feat].copy()

# some macro data are missing first values --> replacing nan with first good value

for feat in macro_feat:

    first_index = df[feat].first_valid_index()

    if first_index != 0:

        df[feat] = df[feat].fillna(df[feat][first_index])

    #print (macro[feat].first_valid_index(),feat,macro[feat].isnull().sum())



# if less than (or equal) 292 columns, we have not merged with macro yet

if train.shape[1] <= 292:

    train = pd.merge(train, df, on='timestamp', how='left')

    test = pd.merge(test, df, on='timestamp', how='left')





plt.figure(figsize=(10,60))

counter = 0

for feat in macro_feat: #['usdrub','brent','mortgage_rate']: 

    counter += 1

    plt.subplot(len(macro_feat),1,counter)

    plt.title(feat, fontsize=15)

    plt.tick_params(labelsize=10)

    plt.plot(df.index, df[feat])

plt.show()
y_train = train["price_doc"]

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)



for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_train[c].values)) 

        x_train[c] = lbl.transform(list(x_train[c].values))

        #x_train.drop(c,axis=1,inplace=True)

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_test[c].values)) 

        x_test[c] = lbl.transform(list(x_test[c].values))





t = time.time()

xgb_params = {'eta': 0.05,'max_depth': 6,'subsample': 1.0,'colsample_bytree': 1.0,

                  'objective': 'reg:linear','eval_metric': 'rmse','silent': 0}

    

                    # 0.05, 5, 0.7, 0.7

                    # reglinear, rmse



dtrain, dtest = xgb.DMatrix(x_train, y_train), xgb.DMatrix(x_test)



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

                       verbose_eval=50, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()



num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)



y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.to_csv('xgbSub.csv', index=False)

output.head()

print ('训练XGboost模型所消耗的时间为:', np.round(time.time() - t, 4),'\n')
featureImportance = model.get_fscore()

features = pd.DataFrame()

features['features'] = featureImportance.keys()

features['importance'] = featureImportance.values()

features.sort_values(by=['importance'],ascending=False,inplace=True)

fig,ax= plt.subplots()

fig.set_size_inches(16,30)

plt.xticks(rotation=60)

sns.set(font_scale=2.5)

sns.barplot(data=features.head(20),x="importance",y="features",ax=ax) #,orient="v")
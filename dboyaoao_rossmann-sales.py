# import required libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import xgboost as xgb

from time import time

import os

print(os.listdir("../input"))
# read data

train = pd.read_csv("../input/train.csv",parse_dates=[2], low_memory=False)

test = pd.read_csv("../input/test.csv",parse_dates=[3], low_memory=False)

store = pd.read_csv("../input/store.csv", low_memory=False)
#train

train.head().append(train.tail())
#test

test.head().append(test.tail())
#store

store.head().append(store.tail())
# check nan

display(train.isnull().sum(),test.isnull().sum(),store.isnull().sum())
# test nan

test[pd.isnull(test.Open)]
# store nan 

store[pd.isnull(store.CompetitionDistance)]
store[pd.isnull(store.CompetitionOpenSinceMonth)].head(10)
NoPW = store[pd.isnull(store.Promo2SinceWeek)]

NoPW[NoPW.Promo2 != 0].shape
# sales of store1 full timne

strain = train[train.Sales>0]

strain.loc[strain['Store']==1 ,['Date','Sales']]     .plot(x='Date',y='Sales',title='Store1',figsize=(16,4))
# sales of store1 ['2014-6-1','2014-7-31']v.s.['2014-8-1','2014-9-30']

strain = train[train.Sales>0]

strain.loc[strain['Store']==1 ,['Date','Sales']]     .plot(x='Date',y='Sales',title='Store1',figsize=(8,2),xlim=['2014-6-1','2014-7-31'])

strain.loc[strain['Store']==1 ,['Date','Sales']]     .plot(x='Date',y='Sales',title='Store1',figsize=(8,2),xlim=['2014-8-1','2014-9-30'])
# the store should be open in the test,so we fillna with 1

test.fillna(1, inplace=True)

#store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace = True)

#store['CompetitionOpenScinceYear'].fillna(store['CompetitionDistance'].median(), inplace = True)

#store['CompetitionOPenScinceMonth'].fillna(store['CompetitionDistance'].median(), inplace = True)
# fillna in store with 0 has better result than median()

store.fillna(0, inplace=True)
display(train.isnull().sum(),test.isnull().sum(),store.isnull().sum())
train = pd.merge(train, store, on='Store')

test = pd.merge(test, store, on='Store')
# split the last 6 weeks data as hold-out set (idea from Gert https://www.kaggle.com/c/rossmann-store-sales/discussion/18024)

train = train.sort_values(['Date'],ascending = False)

ho_test = train[:6*7*1115]

ho_train = train[6*7*1115:]
# only use data of Sales>0 and Open is 1

ho_test = ho_test[ho_test["Open"] != 0]

ho_test = ho_test[ho_test["Sales"] > 0]

ho_train = ho_train[ho_train["Open"] != 0]

ho_train = ho_train[ho_train["Sales"] > 0]
# feature creation (code based on  https://www.kaggle.com/cast42/xgboost-in-python-with-rmspe-v2/code)

def features_create(data):

    

    

    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}

    data.StoreType.replace(mappings, inplace=True)

    data.Assortment.replace(mappings, inplace=True)

    data.StateHoliday.replace(mappings, inplace=True)

    

    

    data['Year'] = data.Date.dt.year

    data['Month'] = data.Date.dt.month

    data['Day'] = data.Date.dt.day

    data['DayOfWeek'] = data.Date.dt.dayofweek

    data['WeekOfYear'] = data.Date.dt.weekofyear

    

    

    

    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) +         (data.Month - data.CompetitionOpenSinceMonth)

    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) +         (data.WeekOfYear - data.Promo2SinceWeek) / 4.0

    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)        

    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)

    

    

  

    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',              7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

    data['monthStr'] = data.Month.map(month2str)

    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''

    data['IsPromoMonth'] = 0

    for interval in data.PromoInterval.unique():

        if interval != '':

            for month in interval.split(','):

                data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1



    return data
features_create(ho_train)

features_create(ho_test)

features_create(test)

print('Features creation finished')
# drop the features of not help

ho_train.drop(['Date','Customers','Open','PromoInterval','monthStr'],axis=1,inplace =True)

ho_test.drop(['Date','Customers','Open','PromoInterval','monthStr'],axis=1,inplace =True)
# the heat map of feature-feature and feature-label correlation 

plt.subplots(figsize=(24,20))

sns.heatmap(ho_train.corr(),annot=True, vmin=-0.1, vmax=0.1,center=0)
ho_xtrain = ho_train.drop(['Sales'],axis=1 )

ho_ytrain = np.log1p(ho_train.Sales)

ho_xtest = ho_test.drop(['Sales'],axis=1 )

ho_ytest = np.log1p(ho_test.Sales)
xtest =test.drop(['Id','Date','Open','PromoInterval','monthStr'],axis = 1)
# define rmspe for xgb(code from https://www.kaggle.com/cast42/xgboost-in-python-with-rmspe-v2/code)

def rmspe(y, yhat):

    return np.sqrt(np.mean((yhat/y-1) ** 2))



def rmspe_xg(yhat, y):

    y = np.expm1(y.get_label())

    yhat = np.expm1(yhat)

    return "rmspe", rmspe(y,yhat)
params = {"objective": "reg:linear",

          "booster" : "gbtree",

          "eta": 0.03,

          "max_depth": 10,

          "subsample": 0.9,

          "colsample_bytree": 0.7,

          "silent": 1,

          "seed": 10

          }

num_boost_round = 6000





dtrain = xgb.DMatrix(ho_xtrain, ho_ytrain)

dvalid = xgb.DMatrix(ho_xtest, ho_ytest)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]



#模型训练

print("Train a XGBoost model")

start = time()

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, 

  early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

end = time()

print('Training time is {:2f} s.'.format(end-start))



#采用保留数据集进行检测

print("validating")

ho_xtest.sort_index(inplace=True) 

ho_ytest.sort_index(inplace=True) 

yhat = gbm.predict(xgb.DMatrix(ho_xtest))

error = rmspe(np.expm1(ho_ytest), np.expm1(yhat))



print('RMSPE: {:.6f}'.format(error))
# analysis by hold-out set

res = pd.DataFrame(data = ho_ytest)

res['Prediction']=yhat

res = pd.merge(ho_xtest,res, left_index= True, right_index=True)

res['Ratio'] = res.Prediction/res.Sales

res['Error'] =abs(res.Ratio-1)

res['Weight'] = res.Sales/res.Prediction

res.head()
col_1 = ['Sales','Prediction']

col_2 = ['Ratio']

L=np.random.randint( low=1,high = 1115, size = 3 ) 

print('Mean Ratio of predition and real sales data is {}: store all'.format(res.Ratio.mean()))

for i in L:

    

    s1 = pd.DataFrame(res[res['Store']==i],columns = col_1)

    s2 = pd.DataFrame(res[res['Store']==i],columns = col_2)

    s1.plot(title = 'Comparation of predition and real sales data: store {}'.format(i),figsize=(12,4))

    s2.plot(title = 'Ratio of predition and real sales data: store {}'.format(i),figsize=(12,4))

    print('Mean Ratio of predition and real sales data is {}: store {}'.format(s2.Ratio.mean(),i))

# the worst 10 samples

res.sort_values(['Error'],ascending=False,inplace= True)

res[:10]
# whole correction

print("weight correction")

W=[(0.990+(i/1000)) for i in range(20)]

S =[]

for w in W:

    error = rmspe(np.expm1(ho_ytest), np.expm1(yhat*w))

    print('RMSPE for {:.3f}:{:.6f}'.format(w,error))

    S.append(error)

Score = pd.Series(S,index=W)

Score.plot()

BS = Score[Score.values == Score.values.min()]

print ('Best weight for Score:{}'.format(BS))
# correction by store

L=range(1115)

W_ho=[]

W_test=[]

for i in L:

    s1 = pd.DataFrame(res[res['Store']==i+1],columns = col_1)

    s2 = pd.DataFrame(xtest[xtest['Store']==i+1])

    W1=[(0.990+(i/1000)) for i in range(20)]

    S =[]

    for w in W1:

        error = rmspe(np.expm1(s1.Sales), np.expm1(s1.Prediction*w))

        S.append(error)

    Score = pd.Series(S,index=W1)

    BS = Score[Score.values == Score.values.min()]

    a=np.array(BS.index.values)

    b_ho=a.repeat(len(s1))

    b_test=a.repeat(len(s2))

    W_ho.extend(b_ho.tolist())

    W_test.extend(b_test.tolist())
yhat_new = yhat*W_ho

error = rmspe(np.expm1(ho_ytest), np.expm1(yhat_new))

print ('RMSPE for weight corretion {:6f}'.format(error))
print("Make predictions on the test set")

dtest = xgb.DMatrix(xtest)

test_probs = gbm.predict(dtest)



# model1  kaggle private score 0.12647

result = pd.DataFrame({"Id": test['Id'], 'Sales': np.expm1(test_probs)})

result.to_csv("Rossmann_submission_1.csv", index=False)



# model2 kaggle private score 0.11756

result = pd.DataFrame({"Id": test['Id'], 'Sales': np.expm1(test_probs*0.995)})

result.to_csv("Rossmann_submission_2.csv", index=False)



# model3 kaggle private score 0.11292

result = pd.DataFrame({"Id": test['Id'], 'Sales': np.expm1(test_probs*W_test)})

result.to_csv("Rossmann_submission_3.csv", index=False)

# ensemble XGBoost model with best kaggle private score 0.11048(20th), but cost too much time (about 7h)

'''

print("Train an new ensemble XGBoost model")

start = time()

rounds = 10

preds_ho = np.zeros((len(ho_xtest.index), rounds))

preds_test = np.zeros((len(test.index), rounds))

B=[]

for r in range(rounds):

    print('round {}:'.format(r+1))

    

    params = {"objective": "reg:linear",

          "booster" : "gbtree",

          "eta": 0.03,

          "max_depth": 10,

          "subsample": 0.9,

          "colsample_bytree": 0.7,

          "silent": 1,

          "seed": r+1

          }

    num_boost_round = 6000

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, 

                    early_stopping_rounds=100, feval=rmspe_xg, verbose_eval=True)

    

    yhat = gbm.predict(xgb.DMatrix(ho_xtest))

    

    L=range(1115)

    W_ho=[]

    W_test=[]

    for i in L:

        s1 = pd.DataFrame(res[res['Store']==i+1],columns = col_1)

        s2 = pd.DataFrame(xtest[xtest['Store']==i+1])

        W1=[(0.990+(i/1000)) for i in range(20)]

        S =[]

        for w in W1:

            error = rmspe(np.expm1(s1.Sales), np.expm1(s1.Prediction*w))

            S.append(error)

        Score = pd.Series(S,index=W1)

        BS = Score[Score.values == Score.values.min()]

        a=np.array(BS.index.values)

        b_ho=a.repeat(len(s1))

        b_test=a.repeat(len(s2))

        W_ho.extend(b_ho.tolist())

        W_test.extend(b_test.tolist())

    



    yhat_ho = yhat*W_ho

    yhat_test =gbm.predict(xgb.DMatrix(xtest))*W_test

    error = rmspe(np.expm1(ho_ytest), np.expm1(yhat_ho))

    B.append(error)

    preds_ho[:, r] = yhat_ho

    preds_test[:, r] = yhat_test

    print('round {} end'.format(r+1))

    

end = time()

time_elapsed = end-start

print('Training is end')

print('Training time is {} h.'.format(time_elapsed/3600))   

'''
'''

print ('Validating')



bagged_ho_preds1 = preds_ho.mean(axis = 1)

error1 = rmspe(np.expm1(ho_ytest), np.expm1(bagged_ho_preds1))

print('RMSPE for mean: {:.6f}'.format(error1))





R = range(10)   

Mw = [0.20,0.20,0.10,0.10,0.10,0.10,0.10,0.10,0.00,0.00] 

A = pd.DataFrame()

A['round']=R

A['best_score']=B

A.sort_values(['best_score'],inplace = True)

A['weight']=Mw

A.sort_values(['round'],inplace = True)

weight=np.array(A['weight'])

preds_ho_w=weight*preds_ho

bagged_ho_preds2 = preds_ho_w.sum(axis = 1)

error2 = rmspe(np.expm1(ho_ytest), np.expm1(bagged_ho_preds2))

print('RMSPE for weight: {:.6f}'.format(error2))

'''
'''

print("Make predictions on the test set")

#model4

bagged_preds = preds_test.mean(axis = 1)

result = pd.DataFrame({"Id": test['Id'], 'Sales': np.expm1(bagged_preds)})

result.to_csv("Rossmann_submission_4.csv", index=False)

# modle5

bagged_preds = (preds_test*weight).sum(axis = 1)

result = pd.DataFrame({"Id": test['Id'], 'Sales': np.expm1(bagged_preds)})

result.to_csv("Rossmann_submission_5.csv", index=False)

'''
# feature importance

xgb.plot_importance(gbm)
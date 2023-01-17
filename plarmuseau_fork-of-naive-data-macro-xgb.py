import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime

#now = datetime.datetime.now()



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')

id_test = test.id

#print(train.info(10))

#print(test.info(10))

#print(macro.info(10))

#print(macro[['grp_growth','construction_value','rent_price_2room_eco','real_dispos_income_per_cap_growth','income_per_cap','mortgage_rate']])

#print(macro['income_per_cap']*12*0.5/(macro['mortgage_rate']/100))
x_train=train.merge(macro[['timestamp','salary','mortgage_rate','rent_price_2room_eco','rent_price_2room_bus','unemployment']], left_on='timestamp', right_on='timestamp', how='left')

#merge macrodata
#Scatter Matrix:

top10=x_train[['price_doc','full_sq','floor','max_floor','build_year','prom_part_5000','office_count_1000','cafe_count_1500_price_2500','product_type','salary','mortgage_rate','rent_price_2room_eco','rent_price_2room_bus','unemployment']].sample(2000)

top10 = top10.dropna()

topratio=pd.DataFrame([])

topratio['val_mort_sal']=top10['price_doc']*top10['mortgage_rate']/top10['salary']/100

topratio['rent_sal']=(top10['rent_price_2room_eco'])/top10['salary']*100

topratio['pri_sq']=top10['price_doc']/top10['full_sq']

topratio['pri_floor']=top10['price_doc']/(top10['floor']+1)

topratio['product_type']=top10['product_type']

topratio['price_doc']=top10['price_doc']

import seaborn as sns; 

sns.set(style="ticks", color_codes=True)

g = sns.pairplot(topratio, hue="product_type")
import seaborn as sns

corr = topratio.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True)


y_train = train["price_doc"]

x_test = test.drop(["id"], axis=1)

#x_test = test[["id","timestamp","full_sq","life_sq","floor","build_year","max_floor","kitch_sq","num_room","state"]]

x_train = train.drop(["id" ,"price_doc"], axis=1)

#x_train = train[["id","timestamp","full_sq","life_sq","floor","build_year","max_floor","kitch_sq","num_room","state"]]



#x_train=x_train.merge(macro[['timestamp','cpi','ppi','usdrub','eurrub','brent']], left_on='timestamp', right_on='timestamp', how='left')





#____________  append macro data

x_train=x_train.merge(macro, left_on='timestamp', right_on='timestamp', how='left')

#x_test=x_test.merge(macro[['timestamp','cpi','ppi','usdrub','eurrub','brent']], left_on='timestamp', right_on='timestamp', how='left')

x_test=x_test.merge(macro, left_on='timestamp', right_on='timestamp', how='left')





#_________________  drop empty columns

x_train=x_train.dropna(axis=1, how='all')

x_test=x_test.dropna(axis=1, how='all')

#can't merge train with test because the kernel run for very long time

#child_on_acc_pre_school,modern_education_share,old_education_build_share 

x_train = x_train.drop(["timestamp"], axis=1)

x_test = x_test.drop(["timestamp"], axis=1)

# find mutual columns

traincol=list(x_train.columns.values)

testcol=list(x_test.columns.values)

mutucol=list(set(traincol).intersection(testcol))

# reshape dataframes

x_train=x_train[mutucol]

x_test=x_test[mutucol]





print(x_train.head())

#print(x_test.info(10))

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

        #x_test.drop(c,axis=1,inplace=True)        
xn_train=x_train.convert_objects(convert_numeric=True).fillna(value=0.0)

xn_test=x_test.convert_objects(convert_numeric=True).fillna(value=0.0)

#print(xn_train)

xno_train = (xn_train - xn_train.mean()) / (xn_train.max() - xn_train.min())

xno_test = (xn_test - xn_test.mean()) / (xn_test.max() - xn_test.min())

#print(xno_train)



#------------------

# test first

top=  (  xno_test[:1].dot(xno_train.T) / ( abs(xno_test[:1]).dot(abs(xno_train.T))) ).T

top.columns=['corre']

print(top.max())

print(float(train.iloc[top[top['corre']==top['corre'].max()].index]['price_doc']))



result=[]

for xi in range(len(xno_test)):

    top=  (  xno_test[xi:xi+1].dot(xno_train.T) / ( abs(xno_test[xi:xi+1]).dot(abs(xno_train.T))) ).T

    top.columns=['corre']

    result.append( float( train.iloc[top[top['corre']==top['corre'].max()].index]['price_doc']) )

    

result.to_csv('xgbSub_3.csv', index=False)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=10, early_stopping_rounds=10,

    verbose_eval=10, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plo
xgb_params = {

    'max_depth': 7,

    'min_child_weight': 3,   

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1,

    'nthread':12,

    'silent':1,

}

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=100, early_stopping_rounds=10,

    verbose_eval=6, show_stdv=False)
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
y_predict = model.predict(dtest)

output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.head()
output.to_csv('xgbSub_2.csv', index=False)
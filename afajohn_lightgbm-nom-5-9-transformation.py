import numpy as np

import pandas as pd

import lightgbm as lgbm

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import normalize

pd.options.display.max_columns = 999

%matplotlib inline
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')

full_df = pd.concat([train,test],axis=0)
train.head()
full_df["ord_5a"]=full_df["ord_5"].str[0]

full_df["ord_5b"]=full_df["ord_5"].str[1]
ordinal_features = ['ord_0','ord_3','ord_4','ord_5','ord_5a','ord_5b']
ord1_enc = OrdinalEncoder(categories=[np.array(['Novice','Contributor','Expert','Master','Grandmaster'])])
full_df.ord_1 = ord1_enc.fit_transform(full_df.ord_1.values.reshape(-1,1)).astype(np.int16)
ord2_enc = OrdinalEncoder(categories=[np.array(['Freezing','Cold','Warm','Hot','Boiling Hot','Lava Hot'])])
full_df.ord_2 = ord2_enc.fit_transform(full_df.ord_2.values.reshape(-1,1)).astype(np.int16)
for feat in ordinal_features:

    enc = OrdinalEncoder()

    full_df[feat] = enc.fit_transform(full_df[feat].values.reshape(-1,1)).astype(np.int16)
hex_df = full_df.loc[:,"nom_5":"nom_9"]
hex_1 = lambda x: int(bin(int(x,16))[2:].zfill(36)[:9],2)

hex_2 = lambda x: int(bin(int(x,16))[2:].zfill(36)[9:18],2)

hex_3 = lambda x: int(bin(int(x,16))[2:].zfill(36)[18:27],2)

hex_4 = lambda x: int(bin(int(x,16))[2:].zfill(36)[27:],2)
new_ord_df = pd.DataFrame()

for col in hex_df:

    new_ord_df['%s_1'%col] = hex_df[col].apply(hex_1)

    new_ord_df['%s_2'%col] = hex_df[col].apply(hex_2)

    new_ord_df['%s_3'%col] = hex_df[col].apply(hex_3)

    new_ord_df['%s_4'%col] = hex_df[col].apply(hex_4)
full_df.drop(hex_df.columns,axis=1,inplace=True)
full_df = pd.concat([full_df,new_ord_df],axis=1)
country_dict = {'Finland':[61.924110,25.748152,'europe',2], 

                'Russia':[61.524010,105.318756,'asia',4], 

                'Canada':[56.130367,-106.346771,'asia',3], 

                'Costa Rica':[9.748917,-83.753426,'sa',1], 

                'China':[35.861660,104.195396,'asia',6], 

                'India':[20.593683,78.962883,'na',5]}
country_df = pd.DataFrame()

country_df['lat'] = full_df.nom_3.apply(lambda x: country_dict[x][0])

country_df['lon'] = full_df.nom_3.apply(lambda x: country_dict[x][1])

country_df['continent'] = full_df.nom_3.apply(lambda x: country_dict[x][2])
full_df = pd.concat([full_df,country_df],axis=1)
for feat in full_df.columns:

    if full_df[feat].dtype == 'object':

        print('Encoding ',feat)

        le = LabelEncoder()

        full_df[feat] = le.fit_transform(full_df[feat].values.reshape(-1,1))
cyclic_days = pd.DataFrame()



cyclic_days['day_sin'] = np.sin(2 * np.pi * full_df['day']/7)

cyclic_days['day_cos'] = np.cos(2 * np.pi * full_df['day']/7)



# full_df['month_sin'] = np.sin(2 * np.pi * train['month']/12)

# full_df['month_cos'] = np.cos(2 * np.pi * train['month']/12)
cyclic_months = pd.DataFrame()



cyclic_months['month_sin'] = np.sin(2 * np.pi * full_df['month']/12)

cyclic_months['month_cos'] = np.cos(2 * np.pi * full_df['month']/12)
full_df = pd.concat([full_df,cyclic_days,cyclic_months],axis=1)
drop_cols = ['target','id','bin_0','ord_5','day','month','nom_3']
y = full_df.target[:len(train)]

X = full_df.drop(drop_cols,axis=1)[:len(train)]
lgb_train = lgbm.Dataset(X,label=y)
params = {

        'max_depth':3,

        'objective': 'binary',

        'feature_fraction': 0.2,

        'bagging_fraction': 1,

        'verbose': -1,

        'is_unbalance':False

    }
# model = lgbm.cv(params,lgb_train,num_boost_round=4000,

#                 early_stopping_rounds=30,metrics='auc',

#                 eval_train_metric=True,verbose_eval=10)
model = lgbm.train(params,lgb_train,num_boost_round=3400)
lgbm.plot_importance(model)
test_X = full_df.drop(drop_cols,axis=1)[len(train):]
sub_predictions = model.predict(test_X)
pd.DataFrame({"id": test["id"], "target": sub_predictions}).to_csv("submission.csv", index=False)
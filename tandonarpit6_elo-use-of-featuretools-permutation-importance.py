import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
from matplotlib import pyplot as plt
import gc
print(os.listdir("../input"))
gc.collect()
train=pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')
import featuretools as ft

es= ft.EntitySet(id= 'train')

variable_types={'feature_1':ft.variable_types.Categorical,'feature_2':ft.variable_types.Categorical, 
                'feature_3':ft.variable_types.Categorical, 'target':ft.variable_types.Id}
es= es.entity_from_dataframe(entity_id='train',dataframe= train, index= 'card_id',variable_types= variable_types)

merchants= pd.read_csv('../input/merchants.csv')
merchants= merchants.drop_duplicates(['merchant_id'])
variable_types={'merchant_group_id':ft.variable_types.Id, 'merchant_category_id':ft.variable_types.Id, 
                'subsector_id':ft.variable_types.Categorical,
               'city_id':ft.variable_types.Id,'state_id':ft.variable_types.Id,'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='merchants', dataframe= merchants, index='merchant_id',variable_types= variable_types)
del merchants
gc.collect()

new_merchant_transactions= pd.read_csv('../input/new_merchant_transactions.csv')
new_merchant_transactions= new_merchant_transactions[(new_merchant_transactions['card_id']).isin(train['card_id'])]
variable_types={'card_id':ft.variable_types.Id, 'state_id':ft.variable_types.Id, 'city_id':ft.variable_types.Id,
               'merchant_category_id':ft.variable_types.Id,'merchant_id':ft.variable_types.Id,'subsector_id':ft.variable_types.Id,
              'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='new_merchant_transactions',dataframe= new_merchant_transactions, make_index= True,
                             index='new_merchants_id',time_index='purchase_date',variable_types= variable_types)
del new_merchant_transactions
gc.collect()

historical_transactions= pd.read_csv('../input/historical_transactions.csv')
historical_transactions= historical_transactions[(historical_transactions['card_id']).isin(train['card_id'])]
variable_types={'card_id':ft.variable_types.Id, 'state_id':ft.variable_types.Id, 'city_id':ft.variable_types.Id,
               'merchant_category_id':ft.variable_types.Id,'merchant_id':ft.variable_types.Id,'subsector_id':ft.variable_types.Id,
              'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='historical_transactions',dataframe= historical_transactions, make_index= True,index='historical_id',
                             time_index='purchase_date',variable_types= variable_types)
del historical_transactions
gc.collect()
r_cards_historical= ft.Relationship(es['train']['card_id'],es['historical_transactions']['card_id'])
es= es.add_relationship(r_cards_historical)

r_cards_new_merchants= ft.Relationship(es['train']['card_id'],es['new_merchant_transactions']['card_id'])
es= es.add_relationship(r_cards_new_merchants)

r_merchants_historical= ft.Relationship(es['merchants']['merchant_id'], es['historical_transactions']['merchant_id'])
es= es.add_relationship(r_merchants_historical)

r_merchants_new_merchants= ft.Relationship(es['merchants']['merchant_id'],es['new_merchant_transactions']['merchant_id'])
es= es.add_relationship(r_merchants_new_merchants)
features_train, feature_names_train= ft.dfs(entityset= es, target_entity= 'train', max_depth= 1)

del es
gc.collect()
features_train
import featuretools as ft

es= ft.EntitySet(id= 'test')

variable_types={'feature_1':ft.variable_types.Categorical,'feature_2':ft.variable_types.Categorical, 
                'feature_3':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='test',dataframe= test, index= 'card_id',variable_types= variable_types)

merchants= pd.read_csv('../input/merchants.csv')
merchants= merchants.drop_duplicates(['merchant_id'])
variable_types={'merchant_group_id':ft.variable_types.Id, 'merchant_category_id':ft.variable_types.Id, 
                'subsector_id':ft.variable_types.Categorical,
               'city_id':ft.variable_types.Id,'state_id':ft.variable_types.Id,'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='merchants', dataframe= merchants, index='merchant_id',variable_types= variable_types)
del merchants
gc.collect()

new_merchant_transactions= pd.read_csv('../input/new_merchant_transactions.csv')
new_merchant_transactions= new_merchant_transactions[(new_merchant_transactions['card_id']).isin(test['card_id'])]
variable_types={'card_id':ft.variable_types.Id, 'state_id':ft.variable_types.Id, 'city_id':ft.variable_types.Id,
               'merchant_category_id':ft.variable_types.Id,'merchant_id':ft.variable_types.Id,'subsector_id':ft.variable_types.Id,
              'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='new_merchant_transactions',dataframe= new_merchant_transactions, make_index= True,
                             index='new_merchants_id',time_index='purchase_date',variable_types= variable_types)
del new_merchant_transactions
gc.collect()

historical_transactions= pd.read_csv('../input/historical_transactions.csv')
historical_transactions= historical_transactions[(historical_transactions['card_id']).isin(test['card_id'])]
variable_types={'card_id':ft.variable_types.Id, 'state_id':ft.variable_types.Id, 'city_id':ft.variable_types.Id,
               'merchant_category_id':ft.variable_types.Id,'merchant_id':ft.variable_types.Id,'subsector_id':ft.variable_types.Id,
              'category_2':ft.variable_types.Categorical}
es= es.entity_from_dataframe(entity_id='historical_transactions',dataframe= historical_transactions, make_index= True,index='historical_id',
                             time_index='purchase_date',variable_types= variable_types)
del historical_transactions
gc.collect()
r_cards_historical= ft.Relationship(es['test']['card_id'],es['historical_transactions']['card_id'])
es= es.add_relationship(r_cards_historical)

r_cards_new_merchants= ft.Relationship(es['test']['card_id'],es['new_merchant_transactions']['card_id'])
es= es.add_relationship(r_cards_new_merchants)

r_merchants_historical= ft.Relationship(es['merchants']['merchant_id'], es['historical_transactions']['merchant_id'])
es= es.add_relationship(r_merchants_historical)

r_merchants_new_merchants= ft.Relationship(es['merchants']['merchant_id'],es['new_merchant_transactions']['merchant_id'])
es= es.add_relationship(r_merchants_new_merchants)
features_test, feature_names_test= ft.dfs(entityset= es, target_entity= 'test', max_depth= 1)

del es
gc.collect()
columns_categorical=['MODE(new_merchant_transactions.authorized_flag)', 'MODE(new_merchant_transactions.category_1)', 
              'MODE(new_merchant_transactions.category_3)', 'MODE(new_merchant_transactions.merchant_id)', 
              'MODE(historical_transactions.authorized_flag)', 'MODE(historical_transactions.category_1)', 
              'MODE(historical_transactions.category_3)', 'MODE(historical_transactions.merchant_id)']
Y=features_train['target']

features_train= features_train.drop(columns=['target'])
features_train= features_train.fillna(method= 'bfill')
features_test= features_test.fillna(method= 'bfill')

X= features_train.copy()
features_train.to_csv('features_train.csv',index= False)
features_test.to_csv('features_test.csv',index= False)
from sklearn.preprocessing import LabelEncoder

temp= features_train.append(features_test)
for i in columns_categorical:
    le= LabelEncoder()
    temp[i]= temp[i].astype('str')
    X[i]= X[i].astype('str')
    features_test[i]= features_test[i].astype('str')
    le.fit(temp[i])
    X[i]= le.transform(X[i])
    features_test[i]= le.transform(features_test[i])
from sklearn.model_selection import train_test_split

xtrain,xval,ytrain,yval= train_test_split(X,Y,test_size=0.1)
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

model= lgb.LGBMRegressor()
model.fit(xtrain,ytrain)

print("RMSE of Validation Data using Light GBM: %.2f" % math.sqrt(mean_squared_error(yval,model.predict(xval))))
fig, ax= plt.subplots(figsize=(14,14))
lgb.plot_importance(model, ax= ax)
plt.show()
import eli5
from eli5.sklearn import PermutationImportance
 
perm = PermutationImportance(model).fit(xval,yval)
eli5.show_weights(perm, feature_names = xval.columns.tolist())
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(xval)
shap.summary_plot(shap_values, xval)
from sklearn.feature_selection import SelectFromModel

submission= pd.read_csv('../input/sample_submission.csv')
features_test= features_test.fillna(0)
features_test= features_test.reindex(index= submission['card_id'])

sel= SelectFromModel(perm, threshold= 0.002, prefit= True)
X= sel.transform(X)
features_test= sel.transform(features_test)

print("Modified shape:", X.shape)
model_1= lgb.LGBMRegressor(learning_rate= 0.1, gamma=1)
model_1.fit(X,Y)
ypred= model_1.predict(features_test)

submission['target']=ypred
submission.to_csv('submission.csv', index= False)
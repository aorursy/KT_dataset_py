# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from sklearn import model_selection, preprocessing

from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

#import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import matplotlib.pyplot as plt



color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)

sns.set(style="white", color_codes=True)



pd.set_option('display.max_rows' , 500)
train_house_df = pd.read_csv('../input/train.csv')

test_house_df = pd.read_csv('../input/test.csv')

macro_df = pd.read_csv('../input/macro.csv')
macro_df.shape
train_house_df.shape
test_house_df.shape
label_df = train_house_df['price_doc']



feature_df = train_house_df.drop(['price_doc'], axis=1 )



merge_df = feature_df.append(test_house_df)



imp_column=['full_sq' , 'life_sq' , 'floor' , 'max_floor' , 'num_room' , 'kitch_sq' , 'state' , 'build_year' , 'material']
imp  =  Imputer(missing_values='NaN' ,strategy='most_frequent' , axis=0)



merge_df[['state','material']] = imp.fit_transform(merge_df[['state','material']])
imp  =  Imputer(missing_values='NaN' ,strategy='median' , axis=0)

merge_df[['life_sq','max_floor' , 'num_room','kitch_sq']] = imp.fit_transform(merge_df[['life_sq','max_floor' , 'num_room','kitch_sq']])



imp  =  Imputer(missing_values='NaN' ,strategy='median' , axis=0)



merge_df['floor'] = imp.fit_transform(merge_df['floor'].values.reshape(-1, 1))
merge_df['build_year'][merge_df['build_year'] <= 1600]=2014



merge_df['build_year'][merge_df['build_year'] > 2050]=2014



merge_df['house_age'] = 2020  - merge_df['build_year']



imp_column.append('house_age')
imp  =  Imputer(missing_values='NaN' ,strategy='median' , axis=0)



merge_df['house_age'] = imp.fit_transform(merge_df['house_age'].values.reshape(-1, 1))



imp_column.remove('build_year')
merge_df['full_sq'][merge_df['full_sq']==0]= merge_df['full_sq'].median()

merge_df['max_floor'][merge_df['max_floor']==0]= merge_df['floor']



merge_df['life_sq_ratio']=merge_df['life_sq']/merge_df['full_sq']



imp_column.remove('life_sq')



imp_column.append('life_sq_ratio')



merge_df['floor_ratio']=merge_df['floor']/(merge_df['max_floor']+1)



imp_column.remove('floor')

imp_column.remove('max_floor')

imp_column.append('floor_ratio')



merge_df['kitch_sq_ratio']=merge_df['kitch_sq']/merge_df['full_sq']



imp_column.remove('kitch_sq')

imp_column.append('kitch_sq_ratio')



merge_df['log_full_sq']=np.log1p(merge_df['full_sq'])



imp_column.remove('full_sq')

imp_column.append('log_full_sq')

imp_column.append('id')
imp_column
merge_df.drop(['hospital_beds_raion' , 'cafe_sum_500_min_price_avg' , 'cafe_sum_500_max_price_avg' , 'cafe_avg_price_500'],inplace=True)
macro_cat_columns  = macro_df.select_dtypes(exclude=['float64' , 'int64']).columns



macro_num_columns  = macro_df.select_dtypes(include=['float64' , 'int64']).columns



from sklearn.preprocessing import Imputer



imp  =  Imputer(missing_values='NaN' ,strategy='median' , axis=0)



macro_df[macro_num_columns] = imp.fit_transform(macro_df[macro_num_columns])
remove_column=['full_sq' , 'life_sq' , 'floor' , 'max_floor' , 'num_room' , 'kitch_sq' , 'state' , 'build_year' , 'material']

other_column =  feature_df.columns.drop(remove_column)



cat_columns  = merge_df[other_column].select_dtypes(exclude=['float64' , 'int64']).columns



num_columns  = merge_df[other_column].select_dtypes(include=['float64' , 'int64']).columns
macro_num_columns_required=['oil_urals', 'balance_trade', 'balance_trade_growth', 'eurrub',

       'net_capital_export', 'micex_rgbi_tr', 'micex_cbi_tr', 'deposits_rate',

       'mortgage_value', 'rent_price_3room_bus', 'power_clinics',

       'seats_theather_rfmin_per_100000_cap']



imp  =  Imputer(missing_values='NaN' ,strategy='median' , axis=0)



merge_df[num_columns] = imp.fit_transform(merge_df[num_columns])



merge_df['product_type'][merge_df['product_type'].isnull()]='Investment'



macro_df['child_on_acc_pre_school'][macro_df['child_on_acc_pre_school']=='#!']='18,200'



macro_df['child_on_acc_pre_school'][macro_df['child_on_acc_pre_school'].isnull()]='18,200'



macro_df['modern_education_share'][macro_df['modern_education_share'].isnull()]='93,17'



macro_df['old_education_build_share'][macro_df['old_education_build_share'].isnull()]='18,95'
num_columns_required = ['green_part_500', 'prom_part_500', 'office_sqm_500', 'trc_count_500',

       'trc_sqm_500', 'cafe_sum_500_min_price_avg',

       'cafe_count_500_price_1000', 'cafe_count_500_price_4000',

       'cafe_count_500_price_high', 'mosque_count_500', 'leisure_count_500',

       'sport_count_500', 'market_count_500', 'prom_part_1000',

       'office_sqm_1000', 'cafe_sum_1000_min_price_avg',

       'cafe_count_1000_price_high', 'mosque_count_1000', 'market_count_1000',

       'trc_sqm_1500', 'cafe_sum_1500_min_price_avg', 'mosque_count_1500',

       'cafe_sum_2000_min_price_avg', 'mosque_count_2000', 'market_count_2000',

       'mosque_count_3000', 'prom_part_5000', 'mosque_count_5000', 'female_f',

       '7_14_female', 'build_count_1971-1995', 'green_zone_km',

       'water_treatment_km', 'water_km', 'big_road1_km', 'railroad_km',

       'fitness_km', 'additional_education_km', 'church_synagogue_km',

       'catering_km']
main_merge_df  =  merge_df[imp_column]



main_merge_df[cat_columns] = merge_df[cat_columns]



main_merge_df[num_columns] = merge_df[num_columns]

#main_merge_df[num_columns_required] = merge_df[num_columns_required]
#macro_col_required=macro_num_columns_required+['child_on_acc_pre_school' , 'modern_education_share' , 'old_education_build_share']

macro_col_required=macro_num_columns_required+macro_cat_columns.tolist()





merge_macro_df = pd.merge(main_merge_df, macro_df[macro_col_required], on='timestamp') 
from sklearn.preprocessing import LabelEncoder



def createDummy(df , var_mod):

    le = LabelEncoder()

    #var_mod = ['PROD_ABBR','STATE_ABBR' ]

    le = LabelEncoder()

    for i in var_mod:

        df[i] = le.fit_transform(df[i])



    #One Hot Coding:

    #df = pd.get_dummies(df, columns=var_mod)

    return df


from sklearn.feature_selection import RFECV

from sklearn.cross_validation import train_test_split



all_feature_columns = merge_macro_df.columns.tolist()



all_feature_columns.remove('timestamp')



rowid=merge_macro_df['id'] 



model_house_df = merge_macro_df[all_feature_columns]



model_house_df['id']=merge_macro_df['id']



cat_columns_model  = model_house_df.select_dtypes(exclude=['float64' , 'int64']).columns



num_columns_model = model_house_df.select_dtypes(include=['float64' , 'int64']).columns



model_house_df['product_type'][model_house_df['product_type'].isnull()]='Investment'

print(model_house_df.shape)



var_mod = cat_columns_model.tolist()

model_house_df = createDummy(model_house_df,var_mod)

print("After Dummy Coding Shape"+str(model_house_df.shape))



train_sequence = 30470

train_house_model_df  = model_house_df.loc[0:train_sequence]



print(train_house_model_df.shape)





#train_house_model_df['price_doc'] = np.log1p(label_df.values)

test_house_df  = model_house_df.loc[(train_sequence+1):len(model_house_df)]



print(test_house_df.shape)

import xgboost as xgb

target = 'price_doc'

IDcol = ['id']

index=0

removeColumn =[ ]

predictors = [x for x in train_house_model_df.columns if x not in [target]+IDcol+removeColumn]



xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(train_house_model_df[predictors], label_df, feature_names=predictors)

dtest = xgb.DMatrix(test_house_df[predictors], feature_names=predictors)
num_rounds = 100

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_rounds)
dtrain_predictions = model.predict(dtrain)

#print('predictors '+str(predictors))



#Print model report:

print ("\nModel Report")

print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(label_df.values, dtrain_predictions)))
# plot the important features #

from xgboost import plot_importance



fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=60, height=0.8, ax=ax)

plt.show()
feat_imp = pd.Series(model.get_fscore())
selected_alg_columns = feat_imp.sort_values(ascending=False).index[0:70].tolist()
selected_alg_columns
dtrain = xgb.DMatrix(train_house_model_df[selected_alg_columns], label_df, feature_names=selected_alg_columns)

dtest = xgb.DMatrix(test_house_df[selected_alg_columns], feature_names=selected_alg_columns)

from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBRegressor

param_test2 = {

'subsample':[i/10.0 for i in range(6,10)],

 'colsample_bytree':[i/10.0 for i in range(6,10)]

 }

gsearch2 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.05, n_estimators=150, max_depth=10,

 min_child_weight=6, subsample=0.7, colsample_bytree=0.7,

 objective= 'reg:linear', seed=27), 

 param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs=4,iid=False, cv=5)

gsearch2.fit(train_house_model_df[selected_alg_columns],label_df)

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#Print model report:

xgb_params = {

    'eta': 0.05,

    'max_depth': 10,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1,

    'min_child_weight':6

    }

num_rounds=500

dtrain = xgb.DMatrix(train_house_model_df[selected_alg_columns], label_df, feature_names=selected_alg_columns)

dtest = xgb.DMatrix(test_house_df[selected_alg_columns], feature_names=selected_alg_columns)



CV_model = xgb.train(dict(xgb_params, silent=0), dtrain, num_rounds)

dtrain_predictions = CV_model.predict(dtrain)

print ("\nModel Report")

print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(label_df.values, dtrain_predictions)))
y_predict = CV_model.predict(dtest)

output = pd.DataFrame({'id': test_house_df['id'].astype(int), 'price_doc': y_predict})

output.head()



output.to_csv('Sub_feat_try_1.csv', index=False)
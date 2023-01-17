# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



# Tools

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error as MSE

from sklearn.metrics import r2_score

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA

from sklearn.decomposition import FastICA

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb



from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import Lasso

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
object_list = train.select_dtypes(include=['object']).columns

display(train[object_list].sample(10).T)

for f in object_list:

    print('Unique in column ',f,' is -> ',len(train[f].unique()))

float_list = train.select_dtypes(include=['float64']).columns

display(train[float_list].sample(10).T)

int_list = train.select_dtypes(include=['int64']).columns

one_columns=[]

for f in int_list:

    if len(train[f].unique())==1:

        one_columns.append(f)

train.drop(columns = one_columns , inplace = True)

test.drop(columns = one_columns , inplace = True)
for f in object_list:

    le = LabelEncoder()

    le.fit(list(train[f].values) + list(test[f].values))

    train[f] = le.transform(list(train[f].values))

    test[f]  = le.transform(list(test[f].values))

Y = train['y']

train.drop(columns=['y'] , inplace = True , axis=1)

combine = pd.concat([train ,  test])

# object_list = object_list[1:]

for f in object_list:

    temp = pd.get_dummies(combine[f])

    combine = pd.concat([combine,temp] , axis =1)

#     combine = combine.drop([f] , axis=1)

train=combine[:train.shape[0]]

test=combine[train.shape[0]:] 
print(train.shape)

print(test.shape)

print(Y.shape)

train_columns = train.columns
def df_column_uniquify(df):

    df_columns = df.columns

    new_columns = []

    for item in df_columns:

        counter = 0

        newitem = item

        while newitem in new_columns:

            counter += 1

            newitem = "{}_{}".format(item, counter)

        new_columns.append(newitem)

    df.columns = new_columns

    return df



train = df_column_uniquify(train)  

test = df_column_uniquify(test)   

train['y']=Y
original_col = list(test.drop(columns=object_list).columns)

display(train.head())

display(test.head())
def get_additional_features(train , test , ID = False):

    col = list(test.columns)

    if ID!= True:

        col.remove('ID')

    n_comp = 12

    #TSVD

    tsvd = TruncatedSVD(n_components = n_comp  , random_state = 98)

    tsvd_result_train = tsvd.fit_transform(train[col])

    tsvd_result_test = tsvd.transform(test[col])

    #PCA

    pca = PCA(n_components = n_comp , random_state = 98)

    pca_result_train = pca.fit_transform(train[col])

    pca_result_test = pca.transform(test[col])

    #FICA

    ica = FastICA(n_components =n_comp , random_state = 98)

    ica_result_train = ica.fit_transform(train[col])

    ica_result_test = ica.transform(test[col])

    #GRP

    grp = GaussianRandomProjection(n_components = n_comp , random_state = 98)

    grp_result_train = grp.fit_transform(train[col])

    grp_result_test = grp.transform(test[col])

    #SRP

    srp = SparseRandomProjection(n_components = n_comp , random_state = 98 , dense_output =True )

    srp_result_train = srp.fit_transform(train[col])

    srp_result_test = srp.transform(test[col])

    for i in range(1,n_comp+1):

        train['tsvd_' + str(i)] = tsvd_result_train[:, i - 1]

        test['tsvd_' + str(i)] = tsvd_result_test[:, i - 1]

        train['pca_' + str(i)] = pca_result_train[:, i - 1]

        test['pca_' + str(i)] = pca_result_test[:, i - 1]

        train['ica_' + str(i)] = ica_result_train[:, i - 1]

        test['ica_' + str(i)] = ica_result_test[:, i - 1]

        train['grp_' + str(i)] = grp_result_train[:, i - 1]

        test['grp_' + str(i)] = grp_result_test[:, i - 1]

        train['srp_' + str(i)] = srp_result_train[:, i - 1]

        test['srp_' + str(i)] = srp_result_test[:, i - 1]

    return train ,test

def get_lgb_data(train , test , col , label , params ,rounds):

    i=0

    RMSE = []

    R2_Score = []

    ID = []

    kf = KFold(n_splits = 5 , shuffle = False)

    train = train.reset_index(drop = True)

    for train_index , test_index in kf.split(train):

        train_x , test_x = train.iloc[train_index, :] ,test.iloc[test_index ,:]

        train_y , test_y = label.iloc[train_index] , label.iloc[test_index] 

        train_lgb = lgb.Dataset(train_x[col] , train_y)

        model = lgb.train(params , train_lgb , num_boost_round = rounds)

        pred = model.predict(test_x[col])

        test_x['label'] = list(test_y)

        test_x['predicted'] = pred

        r2 = r2_score(test_y , pred)

        rmse = MSE(test_y  ,pred)**0.5

        print('R2 Scored of Fold '+str(i+1)+' is '+str(r2))

        R2_Score.append(r2)

        RMSE.append(rmse)

        print('RMSE of Fold '+str(i+1)+' is '+str(rmse))

        ID.append(test_x['ID'])

        if i==0:

            Final = test_x

        else:

            Final = Final.append(test_x,ignore_index=True)

        i+=1

    lgb_train = lgb.Dataset(train[col], label)

    model = lgb.train(params , lgb_train , num_boost_round = rounds)

    pred = model.predict(test[col])

    lgb.plot_importance(model, max_num_features = 20)

    Final_pred = pd.DataFrame({'ID': test['ID'] ,'y':pred})

    print('Out of Bag R2 Score')

    print(np.mean(r2))

    print('Out of Bag RMSE')

    print(np.mean(RMSE))

    return Final , Final_pred
def get_sklearn_data(train , test , model , label ,col):

    ID = []

    RMSE = []

    R2_Score = []

    i=0

    train =train.reset_index(drop = True)

    kf = KFold(n_splits = 5 , shuffle = False)

    for train_index , test_index in kf.split(train):

        train_x , test_x = train.iloc[train_index,:] , train.iloc[test_index,:]

        train_y , test_y = label.iloc[train_index] , label.iloc[test_index]

        model.fit(train_x[col],train_y)

        pred = model.predict(test_x[col])

        test_x['label'] = list(test_y)

        test_x['predicted'] = pred

        r2 = r2_score(test_y , pred)

        rmse = MSE(test_y , pred)**0.5

        print('R2 Scored of Fold '+str(i+1)+' is '+str(r2))

        R2_Score.append(r2)

        RMSE.append(rmse)

        print('RMSE of Fold '+str(i+1)+' is '+str(rmse))

        ID.append(test_x['ID'])

        if i==0:

            Final = test_x

        else:

#             Final = pd.concat([Final , test_x] , axis=0 ) 

            Final = Final.append(test_x,ignore_index=True)

        i+=1

    model.fit(train[col] , label)

    Final_pred = model.predict(test[col])  

    Final_pred = pd.DataFrame({'ID':test['ID'] , 'y':Final_pred})

    print('In of Bag R2 Score')

    print(r2_score(label , model.predict(train[col])))

    print('Out of Bag R2 Score')

    print(np.mean(R2_Score))

    print('In of Bag MSE Score')

    print(MSE(label , model.predict(train[col])))

    print('Out of Bag MSE Score')

    print(np.mean(RMSE))

    return Final , Final_pred , model
def magic_function(train ,test , columns):

    for col in columns:

        magic = train[['ID' , col , 'y']]

        magic = magic.groupby([col])['y'].mean()

        col_name = str(col)+'_mean'

        magic = pd.DataFrame({col: magic.index , col_name:list(magic)})

        magic_mean = magic[col_name].mean()

        train = train.merge(magic , how = 'left' , on = col)

        test = test.merge(magic , how = 'left' , on = col)

        test[col_name] = test[col_name].fillna(magic_mean)

    return train , test
train, test = magic_function(train , test , object_list)

train.sample(5)

train_new , test_new = get_additional_features(train.drop(columns = object_list) ,test.drop(columns = object_list))

train_new = train_new.sample(frac = 1 , random_state = 98)

train_new.sample(10)



train_corr = train_new[original_col].corr().abs()*100

train_corr = train_corr.where(np.triu(np.ones(train_corr.shape)).astype(np.bool))

train_corr.values[[np.arange(train_corr.shape[0])]*2] = np.nan

print(train_corr.shape)

# display(train_corr.tail(20))

counter =0

columns_drop =[]

train_corr_matrix = train_corr.values

for i in range(1 , len(original_col) ,1 ):

    for j in range(i , len(original_col) , 1):

        if train_corr_matrix[i][j] >= 95:

            counter+=1

            columns_drop.append(train_corr.columns[j])

            if counter%20 ==0:

                print('Comman Columns pair reached ... ' , counter)

print(' Total Common Pair Found .... ',counter)

columns_drop = list(set(columns_drop))

# counter =0

# for i in original_col:

#     for j in original_col:

#         if train_corr.loc[[i],[j]].values ==100:

#             counter+=1

#             if counter%10==0:

#                 print('Comman Columns pair ...',counter)

# print(counter)

# train_corr = train_corr.unstack() 

# train_corr = pd.DataFrame(train_corr)

# # display(train_corr.head(10))

# train_corr = train_corr.reset_index()

# train_corr.columns = [['Row' , 'Column' , 'Value']]

# train_corr.dropna(inplace=True)

# # train_corr = train_corr.sort_values(by=['Value'] ,ascending = False)

# # train_corr.loc[train_corr['Value']>60]

# train_corr.reset_index(drop= True , inplace = True)

# train_corr.sample(10)

# train_corr['Value']

# train_corr = pd.DataFrame({'Row':train_corr['Row'].values , 'Column':train_corr['Column'].values ,'Value':  train_corr['Value'].values} )



train_new.drop(columns=columns_drop , inplace = True )

test_new.drop(columns = columns_drop , inplace = True)

col = list(test_new.columns)
%%time

gb1 = GradientBoostingRegressor(n_estimators = 1000 , max_features= 0.95, learning_rate = 0.95 , max_depth = 4)

gb1_train , gb1_test  , model = get_sklearn_data(train_new , test_new , gb1  , train_new['y'] , col)

importances = model.feature_importances_

dataframe = pd.DataFrame({'col':col , 'importance':importances})

dataframe = dataframe.sort_values(by=['importance'] ,ascending = False)

dataframe['importance_ratio'] = dataframe['importance']/dataframe['importance'].max()*100

dataframe = dataframe.head(25)

plt.figure(figsize=(18,6))

plt.barh(dataframe['col'], dataframe['importance_ratio'], color='orange' , align='center' ,linewidth =30 )

plt.xticks(rotation=30)

plt.show()
%%time

lasso1 = Lasso(alpha = 5 , random_state = 98)

lasso_train , lasso_test , model = get_sklearn_data(train_new , test_new , lasso1 , train_new['y'] , col)

%%time

rfr = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

rfr_train , rfr_test , model = get_sklearn_data(train_new , test_new , rfr , train_new['y'] , col)

importances = model.feature_importances_

dataframe = pd.DataFrame({'col':col , 'importance':importances})

dataframe = dataframe.sort_values(by=['importance'] ,ascending = False)

dataframe['importance_ratio'] = dataframe['importance']/dataframe['importance'].max()*100

dataframe = dataframe.head(25)

dataframe['col'] = dataframe['col'].apply(lambda x:  str(x))

plt.figure(figsize=(18,6))

plt.barh(dataframe['col'], dataframe['importance_ratio'], color='orange' , align='center' ,linewidth =30 )

plt.xticks(rotation=30)

plt.show()

print(dataframe.info())
%%time

params = {

            'objective': 'regression',

            'metric': 'rmse',

            'boosting': 'gbdt',

            'learning_rate': 0.0045 ,

            'verbose': 0,

            'num_iterations': 500,

            'bagging_fraction': 0.95,

            'bagging_freq': 1,

            'bagging_seed': 42,

            'feature_fraction': 0.95,

            'feature_fraction_seed': 42,

            'max_bin': 100,

            'max_depth': 3,

            'num_rounds': 800

        }

lgb_train, lgb_test = get_lgb_data(train_new, test_new , col , train_new['y'] , params , 800)
train_new , test_new = get_additional_features(train.drop(columns = object_list) ,test.drop(columns = object_list) , ID = True)

train_new = train_new.sample(frac = 1 , random_state = 98)

train_new.drop(columns=columns_drop , inplace = True )

test_new.drop(columns = columns_drop , inplace = True)

col = list(test_new.columns)

y_mean = np.mean(train_new['y'])

xgb_params = {

        'n_trees': 520, 

        'eta': 0.0045,

        'max_depth': 4,

        'subsample': 0.93,

        'eval_metric': 'rmse',

        'base_score': y_mean, 

        'silent': True,

        'seed': 42,

    }

dtrain = xgb.DMatrix(train_new[col], train_new.y)

dtrain_test = xgb.DMatrix(train_new[col])

dtest = xgb.DMatrix(test_new[col])

    

num_boost_rounds = 1250

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)
stack_train =  gb1_train[['label' , 'predicted']]

stack_train.columns = [['label','gbrp']]

stack_train['lgb'] = lgb_train['predicted']

stack_train['lasso'] = lasso_train['predicted'] 

# stack_train['xgb'] = list(y_pred_train)

# stack_train['rfr'] = rfr_train['predicted']



stack_test = gb1_test[['ID' , 'y']]

stack_test.columns = [['ID' , 'gbrp']]

stack_test['lgb'] = lgb_test['y']

stack_test['lasso'] = lasso_test['y']

# stack_test['xgb'] = list(y_pred)

# stack_test['rfr'] = rfr_test['y']

stack_test = stack_test.drop(['ID'] ,axis = 1)

col =  list(stack_test.columns)







params = {

    'eta': 0.005,

    'max_depth': 2,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean,

    'silent': 1

}



dtrain = xgb.DMatrix(stack_train[col], stack_train['label'])

dtest = xgb.DMatrix(stack_test[col])



xgb_cvalid = xgb.cv(params, dtrain, num_boost_round=2000, early_stopping_rounds=20, verbose_eval=50, show_stdv=True,seed=42)

xgb_cvalid[['train-rmse-mean', 'test-rmse-mean']].plot()

print('Performance does not improve from '+str(len(xgb_cvalid))+' rounds')

model = xgb.train(params,dtrain,num_boost_round =900)

pred_1 = model.predict(dtest)

xgb.plot_importance(model, max_num_features = 3)

plt.show()
Average = 0.70*y_pred + 0.30*pred_1

# Average = np.expm1(Average)

sub = pd.DataFrame({'ID':test['ID'],'y':Average})

sub1 = pd.DataFrame({'ID':test['ID'],'y':y_pred})

sub2 = pd.DataFrame({'ID':test['ID'],'y':pred_1})
# sub.to_csv('Fifth_submission.csv',index=False)

# sub1.to_csv('Sixth_submission.csv',index=False)

sub2.to_csv('12th_submission.csv',index=False)
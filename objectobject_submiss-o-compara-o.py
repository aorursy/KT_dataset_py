# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import gc

from pathlib import Path

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('../'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



# copy-paste

import numpy as np

import pandas as pd

import datetime

import gc

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import TruncatedSVD

import warnings

import math

warnings.filterwarnings('ignore')

np.random.seed(4590)
%%time

PATH = Path('../input/elo-merchant-category-recommendation')

df_train = pd.read_csv(PATH/'train.csv', parse_dates=['first_active_month']);

df_test = pd.read_csv(PATH/'test.csv', parse_dates=['first_active_month']);
%%time

df_hist_trans = pd.read_csv(PATH/'historical_transactions.csv', parse_dates=['purchase_date']);

df_new_merch_trans = pd.read_csv(PATH/'new_merchant_transactions.csv', parse_dates=['purchase_date']);
# https://www.kaggle.com/fabiendaniel/elo-world

def reduce_mem_usage(df, verbose=True):

    prefixes = ['int', 'float']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = str(df[col].dtype)

        if not col_type.startswith('int') and not col_type.startswith('float'):

#             print('col_type:', col_type, 'not compressed')

            continue

        c_min = df[col].min()

        c_max = df[col].max()

        if col_type.startswith('int'):

            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                df[col] = df[col].astype(np.int8)

            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                df[col] = df[col].astype(np.int16)

            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                df[col] = df[col].astype(np.int32)

            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                df[col] = df[col].astype(np.int64)  

        elif col_type.startswith('float'):

            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                df[col] = df[col].astype(np.float16)

            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                df[col] = df[col].astype(np.float32)

            else:

                df[col] = df[col].astype(np.float64)    

    if verbose:

        end_mem = df.memory_usage().sum() / 1024**2

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699/

# Same logic

def fillna_mode(df, cols, pipeline=False):

    for c in cols:

        df[c].fillna(df[c].mode()[0], inplace=True)

    return df if pipeline else None





def get_nan_col_names(df: pd.DataFrame, df_name='<not named>'):

    total = df.shape[0]

    missing_cols = []

    for c in df.columns:

        quo = (total - pd.notna(df[c]).sum())/total

        if quo != 0:

            missing_cols.append(c)

    print(df_name, 'MISSING COLS:', missing_cols)

    return missing_cols
reduce_mem_usage(df_train);

reduce_mem_usage(df_test);

reduce_mem_usage(df_hist_trans);

reduce_mem_usage(df_new_merch_trans);
#https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244

def add_month_diff(df, pipeline=False):

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days) // 30

    df['month_diff'] += df['month_lag']

    return df if pipeline else None





# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699

def pre_process_trans(df, date_col, date_formated=False, add_month=True):

    fillna_mode(df, cols=get_nan_col_names(df))

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0}).astype(np.int8)

    df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0}).astype(np.int8)    

    df['category_2'] = df['category_2'].astype(np.int8)



    if not date_formated:

        df[date_col] = pd.to_datetime(df[date_col])

    df['year'] = df[date_col].dt.year

    df['weekofyear'] = df[date_col].dt.weekofyear

    df['month'] = df[date_col].dt.month

    df['dayofweek'] = df[date_col].dt.dayofweek

    df['weekend'] = (df[date_col].dt.weekday >= 5).astype(np.int8)

    df['hour'] = df[date_col].dt.hour

    if add_month:

        add_month_diff(df)





# https://www.kaggle.com/chauhuynh/my-first-kernel-3-699/

def get_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]





# Refactor of https://www.kaggle.com/chauhuynh/my-first-kernel-3-699/

def custom_group_by(df: pd.DataFrame, df_name: str, agg_by: dict):

    for col in ['category_2','category_3']:

        df[col+'_mean'] = df.groupby([col])['purchase_amount'].transform('mean')

        aggs[col+'_mean'] = ['mean']    

    new_columns = get_new_columns(df_name, agg_by)

#     print("new_columns: \n", *new_columns, sep='\t\n')

    df_group = df.groupby('card_id').agg(agg_by)

    df_group.columns = new_columns

    df_group.reset_index(drop=False, inplace=True)

    df_group[df_name + '_purchase_date_diff'] = (

        df_group[df_name + '_purchase_date_max']

            - df_group[df_name + '_purchase_date_min']

        ).dt.days

    df_group[df_name + '_purchase_date_average'] = (

        df_group[df_name + '_purchase_date_diff']

            / df_group[df_name + '_card_id_size']

        )

    df_group[df_name + '_purchase_date_uptonow'] = (

        datetime.datetime.today()

        - df_group[df_name + '_purchase_date_max']

        ).dt.days



    return df_group
aggs = {}



# Count number of unique values at each column.

for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:

    aggs[col] = ('nunique',)





aggs['purchase_amount'] = ('sum','max','min','mean','var')

aggs['installments'] = ('sum','max','min','mean','var')



# purchase range

aggs['purchase_date'] = ('max','min')



aggs['month_lag'] = ('max','min','mean','var')

aggs['month_diff'] = ('mean',)



# How many transactions:

# - were on the weekend, and the percentage

# - has category_1 as 1 (binary feature), and the percentage

# - were authorized, and its percentage



aggs['weekend'] = ('sum', 'mean')

aggs['category_1'] = ('sum', 'mean')

aggs['authorized_flag'] = ('sum', 'mean')



# How many purchases each card did?

aggs['card_id'] = ('size', )
_old_train = df_train.copy()

_odl_test = df_test.copy()
%%time

gps = []

for name, df in [('hist', df_hist_trans), ('new_hist', df_new_merch_trans)]:    

    pre_process_trans(df, date_col='purchase_date')

    df_group = custom_group_by(df, name, aggs.copy())

    gps.append(df_group)

    df_train = df_train.merge(df_group, on='card_id', how='left')

    df_test = df_test.merge(df_group, on='card_id', how='left')

prob = [

    'hist_purchase_date_max',

    'hist_purchase_date_min',

    'new_hist_purchase_date_max',

    'new_hist_purchase_date_min',

]



def post_process_df(df):

    global prob

    

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    df['fam_dayofweek'] = df['first_active_month'].dt.dayofweek

    df['fam_weekofyear'] = df['first_active_month'].dt.weekofyear

    df['fam_month'] = df['first_active_month'].dt.month

    df['fam_elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days



    df[prob[:2]] = df[prob[:2]].astype(np.int64) * 1e-9



    

    df['transactions_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']

    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']



    for f in ['feature_1','feature_2','feature_3']:

        order_label = df_train.groupby([f])['outliers'].mean()

        df_train[f] = df_train[f].map(order_label)

        df_test[f] = df_test[f].map(order_label)

    
print(df_train.hist_purchase_date_max.notna().sum())

print(df_train.hist_purchase_date_min.notna().sum())

print(df_train.new_hist_purchase_date_max.notna().sum())

print(df_train.new_hist_purchase_date_min.notna().sum())
df_train['outliers'] = (df_train.target < -30).astype(np.int8)
post_process_df(df_train)

post_process_df(df_test)
df_test.drop(['new_hist_purchase_date_max', 'new_hist_purchase_date_min'], axis=1,inplace=True)

df_train.drop(['new_hist_purchase_date_max', 'new_hist_purchase_date_min'], axis=1,inplace=True)
def Tree_Train(df_train,df_test,target,num_leaves):

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)

    oof = np.zeros(len(df_train))

    predictions_train = np.zeros(len(df_train)) 

    predictions_test = np.zeros(len(df_test))

    feature_importance_df = pd.DataFrame()

    df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'outliers', 'target']]

    clf= RandomForestRegressor(max_leaf_nodes=num_leaves,random_state=3101,criterion='mse',bootstrap=True,oob_score=True)

    print("Fitting Random Forests")

    clf.fit(df_train[df_train_columns].fillna(0),target)

    print("Error (rmse oob_score): "+str(math.sqrt(clf.oob_score_)))

    feature_importance_df = pd.DataFrame(data=zip(df_train_columns,clf.feature_importances_),columns=['Feature','importance'])

    predictions_test = clf.predict(df_test[df_train_columns].fillna(0))

    predictions_train = clf.predict(df_train[df_train_columns].fillna(0))

    rmse=0

    for i,item in enumerate(target):

        rmse+=(item-predictions_train[i])**2

    print("Error (rmse over training set): "+str(math.sqrt(rmse/len(target))))



    return {'predictions_test':predictions_test,'predictions_train':predictions_train,'feature_importances':feature_importance_df}

    
#A dataframe containing only numerical values

df_num=df_train[[col for col in df_train.columns if not col in ['target','outliers']]].select_dtypes(include='number')



#NaN/Missing values will be replaced with zeroes

df_num.fillna(0,inplace=True)



decomp = TruncatedSVD(n_components=len(df_num.columns.values)-1,random_state=1)

decomp.fit(df_num)
plot=sns.lineplot(x=range(10),y=decomp.singular_values_[:10])

plot.set_title("Eigenvalues in descending order")

plot.set(ylabel="Eigenvalue",xlabel="Vector")

plt.show()

plot=sns.lineplot(x=range(3,len(decomp.singular_values_)),y=decomp.singular_values_[3:])

plot.set_title("Eigenvalues [from 3rd on] in descending order")

plot.set(ylabel="Eigenvalue",xlabel="Vector")
sns.heatmap([[x**2 for x in component] for component in decomp.components_[:10]])
for component in decomp.components_[:3]:

    weights=sorted(zip([x**2 for x in component],df_num.columns.values),reverse=True)[:3]

    sns.barplot(data=pd.DataFrame(weights,columns=['Squared Weight','Feature Name']),y='Feature Name',x='Squared Weight')

    plt.show()
for num_vectors in range(1,len(decomp.singular_values_)):

    acc_ratio=sum(decomp.singular_values_[:num_vectors])/sum(decomp.singular_values_)

    if (sum(decomp.singular_values_[:num_vectors])/sum(decomp.singular_values_))>=0.80:

            break;

print(str(num_vectors)+" ("+str(round(acc_ratio*10000)/100)+"%)")
sq_eigenval=dict(zip(df_num.columns.values,[0 for i in range(len(df_num.columns.values))]))

for component in decomp.components_[:num_vectors]:

    weights=zip([x**2 for x in component],df_num.columns.values)

    for line in weights:

        sq_eigenval[line[1]]+=line[0] if not np.isnan(line[0]) else 0



#Normalization

sq_eigenval=pd.DataFrame(sq_eigenval.items(),columns=['Feature','Sq. Weight']).sort_values(by='Sq. Weight')

sq_eigenval['Norm. Sq. W.']=(sq_eigenval['Sq. Weight']-sq_eigenval['Sq. Weight'].min())/(sq_eigenval['Sq. Weight'].max()-sq_eigenval['Sq. Weight'].min())



plt.figure(figsize=(14,25))

sns.barplot(

            y='Feature',

            x='Norm. Sq. W.',

            data=sq_eigenval

)

plt.title('Normalized Sq. Weights of Features (SVD)')

plt.tight_layout()

plt.savefig('svd_importances.png')
df_num=df_train[[col for col in df_train.columns if not col in ['target','outliers']]].select_dtypes(include='number')

df_num.fillna(0,inplace=True)

df_svd_train=pd.DataFrame([x[:num_vectors]for x in decomp.transform(df_num)],columns=['svd_'+str(i) for i in range(num_vectors)])

#df_train_svd=df_train.join(df_svd,lsuffix='_caller', rsuffix='_other')

df_svd_train['target']=df_train['target']

df_svd_train['outliers']=df_train['outliers']

df_num=pd.DataFrame(data=df_test,columns=df_num.columns.values)

df_num.fillna(0,inplace=True)

df_svd_test=pd.DataFrame([x[:num_vectors]for x in decomp.transform(df_num)],columns=['svd_'+str(i) for i in range(num_vectors)])

#df_test_svd=df_test.join(df_svd,lsuffix='_caller', rsuffix='_other')

print("Datasets generated.")
sns.scatterplot(data=df_svd_train,x='svd_1',y='svd_0',hue='target')

plt.show()

sns.scatterplot(data=df_svd_train.query('svd_1>0'),x='svd_1',y='svd_0',hue='target')
target=df_svd_train['target']

del df_svd_train['target']
%%time

results=Tree_Train(df_svd_train,df_svd_test,target,5)

predictions_train_svd=results['predictions_train']

predictions_test_svd=results['predictions_test']
df_train['svd_prediction']=predictions_train_svd

df_test['svd_prediction']=predictions_test_svd



df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'outliers', 'target']]

target = df_train['target']

del df_train['target']
%%time

results=Tree_Train(df_train,df_test,target,31)

feature_importance_df=results['feature_importances']

predictions_brute=results['predictions_test']
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('Random Tree Feature Importances')

plt.tight_layout()

plt.savefig('tree_importances.png')
temporal_features=[c for c in df_train.columns if (("week" in c) or ("day" in c) or ("date" in c) or ("year" in c))]

df_train_temporal = df_train[temporal_features]

df_test_temporal = df_test[temporal_features]

df_train_temporal['outliers']=df_train['outliers']

df_train_atemporal = df_train[[c for c in df_train.columns if c not in temporal_features]]

df_test_atemporal = df_test[[c for c in df_test.columns if c not in temporal_features]]

df_train_atemporal['outliers']=df_train['outliers']
%%time

results=Tree_Train(df_train_atemporal,df_test_atemporal,target,31)

predictions_test_atemporal=results['predictions_test']

predictions_train_atemporal=results['predictions_train']
%%time

results=Tree_Train(df_train_temporal,df_test_temporal,target,31)

predictions_test_temporal=results['predictions_test']

predictions_train_temporal=results['predictions_train']
df_train_impera = df_train[['svd_prediction','outliers']]

df_train_impera['temporal_prediction']=predictions_train_temporal

df_train_impera['atemporal_prediction']=predictions_train_atemporal

df_test_impera=df_test[['svd_prediction']]

df_test_impera['temporal_prediction']=predictions_test_temporal

df_test_impera['atemporal_prediction']=predictions_test_atemporal
%%time

results=Tree_Train(df_train_impera,df_test_impera,target,31)

feature_importance_df=results['feature_importances']

predictions_impera=results['predictions_test']
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(7,3))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('Random Tree Feature Importances for Divide-et-Impera method')

plt.tight_layout()

plt.savefig('tree_dei_importances.png')
df_train['temporal_prediction']=predictions_train_temporal

df_train['atemporal_prediction']=predictions_train_atemporal

df_test['temporal_prediction']=predictions_test_temporal

df_test['atemporal_prediction']=predictions_test_atemporal

df_train=df_train.join(df_svd_train[[c for c in df_svd_train.columns if "svd" in c]], lsuffix='_caller', rsuffix='_other')

df_test=df_test.join(df_svd_test[[c for c in df_svd_train.columns if "svd" in c]], lsuffix='_caller', rsuffix='_other')
%%time

results=Tree_Train(df_train,df_test,target,31)

feature_importance_df=results['feature_importances']

predictions_frankie=results['predictions_test']
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)



best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,25))

sns.barplot(x="importance",

            y="Feature",

            data=best_features.sort_values(by="importance",

                                           ascending=False))

plt.title('Random Tree Feature Importances for Frankenstein\'s method')

plt.tight_layout()

plt.savefig('tree_frankie_importances.png')
sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_frankie

sub_df.to_csv("submission_frankie.csv", index=False)



sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_impera

sub_df.to_csv("submission_impera.csv", index=False)



sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_brute

sub_df.to_csv("submission_brutus.csv", index=False)



sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_test_atemporal

sub_df.to_csv("submission_atemporal.csv", index=False)



sub_df = pd.DataFrame({"card_id":df_test["card_id"].values})

sub_df["target"] = predictions_test_svd

sub_df.to_csv("submission_naive_svd.csv", index=False)
lgbm=pd.DataFrame(data=[

["Only SVD projections",3.81,3.89],

["SVD prediction + Dataset Features Cascade",3.63,3.70],

["Divide-et-Impera Cascade",3.33,3.80],

["Non-time-related Features",3.64,3.71],

["Frankie Cascade",2.87,3.87]],columns=['Feature Set','Validation Error','Test Error'])

plt.title('Error plot for LGB-based models', color='black')

plot=sns.scatterplot(data=lgbm,x="Validation Error",y="Test Error",hue='Feature Set')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
forestsm=pd.DataFrame(data=[

["Only SVD projections",0.008497750292186312,3.9],

["SVD prediction + Dataset Features Cascade",0.00811753831695943,3.76],

["Divide-et-Impera Cascade",0.007901780981265769,3.79],

["Non-time-related Features",0.008128093345136984,3.76],

["Frankie Cascade",0.007733020898929636,3.79]],columns=['Feature Set','Validation Error','Test Error'])

#This correction had to be made because we miscalculated the RMSE in the original version.

forestsm['Validation Error']=np.sqrt(((forestsm['Validation Error']*123623)**2)/123623)

plt.title('Error plot for Random Forest-based models', color='black')

plot=sns.scatterplot(data=forestsm,x="Validation Error",y="Test Error",hue='Feature Set')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.show()
forestsm['Type']='Random Forests'

lgbm['Type']='LGB'

plot=sns.lmplot(data=pd.concat([forestsm,lgbm]).query(expr='`Feature Set`!="Only SVD projections"'),x="Validation Error",y="Test Error",hue='Type')